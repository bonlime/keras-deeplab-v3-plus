from pathlib import Path

from PIL import Image
import numpy as np
from numpy import ndarray

from generate_labels import generate_labels
from pascal_voc_classes import PascalVOCClasses

TEST_IMAGE_FOLDER = Path(__file__).parent.joinpath('test_images').resolve()

LANDSCAPE_TEST_IMAGE_NAME = "dog_baxter.jpg"
PORTRAIT_TEST_IMAGE_NAME = "cat.jpg"
SMALL_TEST_IMAGE_NAME = "baptist400.jpg"
RGBA_TEST_IMAGE_NAME = "rgba.png"

LANDSCAPE_TEST_IMAGE = f"{TEST_IMAGE_FOLDER}/{LANDSCAPE_TEST_IMAGE_NAME}"
PORTRAIT_TEST_IMAGE = f"{TEST_IMAGE_FOLDER}/{PORTRAIT_TEST_IMAGE_NAME}"
SMALL_TEST_IMAGE = f"{TEST_IMAGE_FOLDER}/{SMALL_TEST_IMAGE_NAME}"
RGBA_TEST_IMAGE = f"{TEST_IMAGE_FOLDER}/{RGBA_TEST_IMAGE_NAME}"


def test_generate_labels_returns_labels_as_array():
    input_image = Image.open(LANDSCAPE_TEST_IMAGE)
    img_array = np.array(input_image)

    result_array = generate_labels(img_array)

    assert type(result_array) == ndarray


def test_generate_labels_landscape_image_returns_labels_same_dimensions_as_original():
    input_image = Image.open(LANDSCAPE_TEST_IMAGE)
    img_array = np.array(input_image)

    result_array = generate_labels(img_array)
    result_image = Image.fromarray(result_array)

    assert result_image.size == input_image.size


def test_generate_labels_landscape_image_labels_dog_and_background():
    input_image = Image.open(LANDSCAPE_TEST_IMAGE)
    img_array = np.array(input_image)

    result_array = generate_labels(img_array)
    unique_labels = np.unique(result_array)

    assert sorted(unique_labels) == [PascalVOCClasses.background.value, PascalVOCClasses.dog.value]


def test_generate_labels_portrait_image_returns_labels_same_dimensions_as_original():
    input_image = Image.open(PORTRAIT_TEST_IMAGE)
    img_array = np.array(input_image)

    result_array = generate_labels(img_array)
    result_image = Image.fromarray(result_array)

    assert result_image.size == input_image.size


def test_generate_labels_landscape_image_labels_cat_and_background():
    input_image = Image.open(PORTRAIT_TEST_IMAGE)
    img_array = np.array(input_image)

    result_array = generate_labels(img_array)
    unique_labels = np.unique(result_array)

    assert PascalVOCClasses.background.value in unique_labels
    assert PascalVOCClasses.cat.value in unique_labels


def test_generate_labels_size_less_than_default_returns_correct_size():
    input_image = Image.open(SMALL_TEST_IMAGE)
    img_array = np.array(input_image)

    result_array = generate_labels(img_array)
    result_image = Image.fromarray(result_array)

    assert result_image.size == input_image.size


def test_generate_labels_rgba_image_returns_labels_as_array():
    input_image = Image.open(RGBA_TEST_IMAGE)
    img_array = np.array(input_image)

    result_array = generate_labels(img_array)

    assert type(result_array) == ndarray
