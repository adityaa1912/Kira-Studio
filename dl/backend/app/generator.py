import base64
import io
import os
import threading
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


LATENT_DIM = 32
IMAGE_SIZE = 64
OUTPUT_SIZE = 256
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
TRAINED_MODEL_PATH = MODEL_DIR / "animeface-generator.keras"
PRETRAINED_MODEL_PATH = MODEL_DIR / "animeface_generator_model.h5"
UPSCALE_MODEL_PATH = MODEL_DIR / "super-resolution-10.onnx"
MODEL_PATH = Path(
    os.getenv(
        "ANIME_GENERATOR_MODEL",
        str(PRETRAINED_MODEL_PATH),
    )
)
_generator = None
_upscaler = None
_load_lock = threading.Lock()
_generation_lock = threading.Lock()
_upscale_load_lock = threading.Lock()
UPSCALED_SIZE = 672


def _build_generator():
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(LATENT_DIM,)),
            tf.keras.layers.Dense(16 * 16 * 128),
            tf.keras.layers.Reshape((16, 16, 128)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(
                64, kernel_size=5, strides=2, padding="same", activation="selu"
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(
                3, kernel_size=5, strides=2, padding="same", activation="tanh"
            ),
        ]
    )


def get_generator():
    global _generator
    if _generator is not None:
        return _generator

    with _load_lock:
        if _generator is not None:
            return _generator
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Generator model not found: {MODEL_PATH}")

        if MODEL_PATH.suffix.lower() in {".h5", ".hdf5"}:
            model = _build_generator()
            model.load_weights(MODEL_PATH)
        else:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)

        output_shape = model.output_shape
        if tuple(output_shape[-3:]) != (IMAGE_SIZE, IMAGE_SIZE, 3):
            raise ValueError(f"Expected a 64x64 RGB generator, got output shape {output_shape}")
        _generator = model
        return _generator


def get_upscaler():
    global _upscaler
    if _upscaler is not None:
        return _upscaler

    with _upscale_load_lock:
        if _upscaler is not None:
            return _upscaler
        if not UPSCALE_MODEL_PATH.is_file():
            raise FileNotFoundError(f"AI upscaler model not found: {UPSCALE_MODEL_PATH}")
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError(
                "AI upscaling requires onnxruntime; install backend requirements."
            ) from error
        try:
            options = ort.SessionOptions()
            options.log_severity_level = 3
            _upscaler = ort.InferenceSession(
                str(UPSCALE_MODEL_PATH),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise RuntimeError(f"Unable to load AI upscaler: {error}") from error
        return _upscaler


def _upscale_image(image):
    session = get_upscaler()
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_image = image.resize((224, 224), Image.Resampling.LANCZOS)
    y_channel, cb_channel, cr_channel = input_image.convert("YCbCr").split()
    model_input = np.asarray(y_channel, dtype=np.float32)[None, None, ...] / 255.0
    output = session.run([output_name], {input_name: model_input})[0][0, 0]
    output_y = Image.fromarray(np.clip(output * 255.0, 0, 255).astype(np.uint8))
    output_size = output_y.size
    enhanced = Image.merge(
        "YCbCr",
        (
            output_y,
            cb_channel.resize(output_size, Image.Resampling.BICUBIC),
            cr_channel.resize(output_size, Image.Resampling.BICUBIC),
        ),
    ).convert("RGB")
    if enhanced.size != (UPSCALED_SIZE, UPSCALED_SIZE):
        raise RuntimeError(f"Unexpected AI upscaler output dimensions: {enhanced.size}")
    return enhanced


def generate_faces(count, seed=None, variation=1.0, ai_upscale=True):
    model = get_generator()
    with _generation_lock:
        if seed is None:
            noise = tf.random.normal((count, LATENT_DIM)) * variation
        else:
            noise = tf.random.stateless_normal(
                (count, LATENT_DIM), seed=(int(seed), 0)
            ) * variation
        generated = model(noise, training=False).numpy()

    faces = []
    for pixels in generated:
        pixels = np.clip((pixels + 1.0) * 127.5, 0, 255).astype(np.uint8)
        image = Image.fromarray(pixels, mode="RGB")
        image = image.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.Resampling.LANCZOS)
        if ai_upscale:
            image = _upscale_image(image)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95, optimize=True, subsampling=0)
        faces.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
    return faces