import argparse
import csv
import random
from pathlib import Path

import tensorflow as tf
from PIL import Image


IMAGE_SIZE = 64
LATENT_DIM = 32
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def build_generator() -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(LATENT_DIM,)),
            tf.keras.layers.Dense(
                4 * 4 * 256,
                use_bias=False,
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            ),
            tf.keras.layers.Reshape((4, 4, 256)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(
                128, kernel_size=4, strides=2, padding="same",
                use_bias=False,
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(
                64, kernel_size=4, strides=2, padding="same",
                use_bias=False,
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(
                32, kernel_size=4, strides=2, padding="same",
                use_bias=False,
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(
                3, kernel_size=4, strides=2, padding="same",
                activation="tanh",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            ),
        ],
        name="generator",
    )


def build_discriminator() -> tf.keras.Model:
    layers = [
        tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        tf.keras.layers.Conv2D(
            64,
            kernel_size=4,
            strides=2,
            padding="same",
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
        ),
        tf.keras.layers.LeakyReLU(alpha=0.2),
    ]
    for filters in (128, 256, 512):
        layers.extend(
            [
                tf.keras.layers.Conv2D(
                    filters, kernel_size=4, strides=2, padding="same",
                    use_bias=False,
                    kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(alpha=0.2),
            ]
        )
    layers.extend([tf.keras.layers.Flatten(), tf.keras.layers.Dense(1)])
    return tf.keras.Sequential(layers, name="discriminator")


def build_dataset(
    dataset_dir: Path,
    batch_size: int,
    seed: int,
    max_images: int | None = None,
) -> tuple[tf.data.Dataset, int]:
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    image_paths = sorted(
        path for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if max_images is not None:
        if max_images < 1:
            raise ValueError("max_images must be greater than zero")
    if not image_paths:
        raise ValueError(f"No supported images found in {dataset_dir}")

    if max_images is not None and len(image_paths) > max_images:
        image_paths = random.Random(seed).sample(image_paths, max_images)

    dataset = tf.data.Dataset.from_tensor_slices([str(path) for path in image_paths])
    dataset = dataset.shuffle(
        buffer_size=len(image_paths), seed=seed, reshuffle_each_iteration=True
    )

    def preprocess(image_path):
        image = tf.io.read_file(image_path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image.set_shape((None, None, 3))
        image = tf.image.resize_with_pad(
            image, IMAGE_SIZE, IMAGE_SIZE, method="lanczos3", antialias=True
        )
        image.set_shape((IMAGE_SIZE, IMAGE_SIZE, 3))
        image = tf.image.random_flip_left_right(image)
        return image / 127.5 - 1.0

    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset, len(image_paths)


def save_preview(generator: tf.keras.Model, noise: tf.Tensor, output_path: Path) -> float:
    generated = generator(noise, training=False).numpy()
    tile_size = 256
    canvas = Image.new("RGB", (4 * tile_size, 4 * tile_size))
    for index, pixels in enumerate(generated):
        pixels = tf.cast(tf.clip_by_value((pixels + 1.0) * 127.5, 0, 255), tf.uint8).numpy()
        image = Image.fromarray(pixels, mode="RGB")
        image = image.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        row, column = divmod(index, 4)
        canvas.paste(image, (column * tile_size, row * tile_size))
    canvas.save(output_path, optimize=True)
    return float(tf.reduce_mean(tf.math.reduce_std(generated, axis=0)).numpy())


def save_generator(generator: tf.keras.Model, output_path: Path) -> None:
    temporary_path = output_path.with_name(f"{output_path.stem}.tmp.keras")
    generator.save(temporary_path)
    temporary_path.replace(output_path)


def append_metrics(
    metrics_path: Path,
    epoch: int,
    generator_loss: float,
    discriminator_loss: float,
    sample_diversity: float,
) -> None:
    write_header = not metrics_path.exists() or metrics_path.stat().st_size == 0
    with metrics_path.open("a", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=(
                "epoch",
                "generator_loss",
                "discriminator_loss",
                "sample_diversity",
            ),
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "epoch": epoch,
                "generator_loss": f"{generator_loss:.6f}",
                "discriminator_loss": f"{discriminator_loss:.6f}",
                "sample_diversity": f"{sample_diversity:.6f}",
            }
        )


def train(
    dataset_dir: Path,
    output_path: Path,
    epochs: int,
    batch_size: int,
    max_images: int | None = None,
    preview_dir: Path | None = None,
    seed: int = 1337,
    resume: bool = False,
    checkpoint_dir: Path | None = None,
    require_gpu: bool = False,
) -> None:
    if epochs < 1:
        raise ValueError("epochs must be greater than zero")
    if batch_size < 1:
        raise ValueError("batch_size must be greater than zero")
    if not 0 <= seed <= 2_147_483_647:
        raise ValueError("seed must be between 0 and 2,147,483,647")

    gpu_devices = tf.config.list_physical_devices("GPU")
    if require_gpu and not gpu_devices:
        raise RuntimeError(
            "GPU training was required, but TensorFlow detected no compatible GPU. "
            "Run this script with the separate .venv-gpu Python executable."
        )
    if gpu_devices:
        print(f"Using GPU device(s): {', '.join(device.name for device in gpu_devices)}")
    else:
        print("No TensorFlow-compatible GPU detected; training will use CPU.")

    tf.keras.utils.set_random_seed(seed)
    dataset, image_count = build_dataset(dataset_dir, batch_size, seed, max_images)
    generator = build_generator()
    discriminator = build_discriminator()
    optimizer_generator = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    optimizer_discriminator = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)

    output_path = Path(output_path).expanduser().resolve()
    preview_dir = Path(preview_dir).expanduser().resolve() if preview_dir else output_path.parent / "training_previews"
    checkpoint_dir = (
        Path(checkpoint_dir).expanduser().resolve()
        if checkpoint_dir
        else output_path.parent / "checkpoints"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_path.parent / "training_metrics.csv"
    if not resume:
        metrics_path.write_text(
            "epoch,generator_loss,discriminator_loss,sample_diversity\n",
            encoding="utf-8",
        )
    preview_noise = tf.random.stateless_normal((16, LATENT_DIM), seed=(seed, 0))
    completed_epoch = tf.Variable(0, dtype=tf.int64, trainable=False, name="completed_epoch")
    checkpoint = tf.train.Checkpoint(
        generator=generator,
        discriminator=discriminator,
        optimizer_generator=optimizer_generator,
        optimizer_discriminator=optimizer_discriminator,
        completed_epoch=completed_epoch,
    )
    manager = tf.train.CheckpointManager(checkpoint, str(checkpoint_dir), max_to_keep=5)
    if resume:
        if not manager.latest_checkpoint:
            raise FileNotFoundError(f"No training checkpoint found in {checkpoint_dir}")
        checkpoint.restore(manager.latest_checkpoint)
        print(f"Resumed from {manager.latest_checkpoint}")

    @tf.function
    def train_step(real_images):
        current_batch = tf.shape(real_images)[0]
        noise = tf.random.normal((current_batch, LATENT_DIM))
        generated_images = tf.stop_gradient(generator(noise, training=False))

        with tf.GradientTape() as discriminator_tape:
            real_scores = discriminator(real_images, training=True)
            fake_scores = discriminator(generated_images, training=True)
            discriminator_loss = 0.5 * (
                tf.reduce_mean(tf.square(real_scores - 1.0))
                + tf.reduce_mean(tf.square(fake_scores))
            )
        discriminator_gradients = discriminator_tape.gradient(
            discriminator_loss, discriminator.trainable_variables
        )
        optimizer_discriminator.apply_gradients(
            zip(discriminator_gradients, discriminator.trainable_variables)
        )

        noise = tf.random.normal((current_batch, LATENT_DIM))
        with tf.GradientTape() as generator_tape:
            generated_images = generator(noise, training=True)
            generated_scores = discriminator(generated_images, training=False)
            generator_loss = 0.5 * tf.reduce_mean(tf.square(generated_scores - 1.0))
        generator_gradients = generator_tape.gradient(
            generator_loss, generator.trainable_variables
        )
        optimizer_generator.apply_gradients(
            zip(generator_gradients, generator.trainable_variables)
        )
        tf.debugging.assert_all_finite(generator_loss, "Generator loss became non-finite")
        tf.debugging.assert_all_finite(discriminator_loss, "Discriminator loss became non-finite")
        return generator_loss, discriminator_loss

    start_epoch = int(completed_epoch.numpy())
    if start_epoch >= epochs:
        raise ValueError(
            f"Checkpoint already completed {start_epoch} epochs; request more than that to continue"
        )
    print(f"Training on {image_count:,} images for {epochs - start_epoch} epochs (seed {seed})")
    for epoch in range(start_epoch, epochs):
        generator_losses = tf.keras.metrics.Mean()
        discriminator_losses = tf.keras.metrics.Mean()
        for real_images in dataset:
            generator_loss, discriminator_loss = train_step(real_images)
            generator_losses.update_state(generator_loss)
            discriminator_losses.update_state(discriminator_loss)

        generator_value = float(generator_losses.result().numpy())
        discriminator_value = float(discriminator_losses.result().numpy())
        completed_epoch.assign(epoch + 1)
        manager.save()
        preview_path = preview_dir / f"epoch-{epoch + 1:03d}.png"
        diversity_value = save_preview(generator, preview_noise, preview_path)
        append_metrics(
            metrics_path,
            epoch + 1,
            generator_value,
            discriminator_value,
            diversity_value,
        )
        save_generator(generator, output_path)
        print(
            f"Epoch {epoch + 1}/{epochs} | generator {generator_value:.4f} | "
            f"discriminator {discriminator_value:.4f} | "
            f"dataset diversity {diversity_value:.4f}"
        )
        print(f"Saved candidate to {output_path}")
        print(f"Saved preview to {preview_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a reproducible 64x64 DCGAN candidate without replacing the active model."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "animeface-generator-experiment.keras",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "training_previews",
    )
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-images", type=int, help="Train on a bounded dataset subset")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint")
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Stop instead of silently training on CPU when no GPU is detected",
    )
    arguments = parser.parse_args()
    train(
        arguments.dataset,
        arguments.output,
        arguments.epochs,
        arguments.batch_size,
        max_images=arguments.max_images,
        preview_dir=arguments.preview_dir,
        seed=arguments.seed,
        resume=arguments.resume,
        checkpoint_dir=arguments.checkpoint_dir,
        require_gpu=arguments.require_gpu,
    )


if __name__ == "__main__":
    main()
