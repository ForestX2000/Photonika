"""Source disjointness, duplicate protection and memory-mapped dataset caches."""
import numpy as np
import pytest
import zipfile
from PIL import Image

from ml import data


@pytest.fixture
def photo_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "load_skimage", lambda name: None)
    rng = np.random.default_rng(9)
    for index in range(50):
        image = rng.integers(0, 256, (72, 72), dtype=np.uint8)
        Image.fromarray(image).save(tmp_path / f"{index:03}.png")
    return tmp_path


def test_source_splits_are_disjoint_deterministic_and_group_duplicates(photo_directory):
    manifest = data.photo_source_manifest(photo_directory)
    assert {row["split"] for row in manifest} == set(data.PHOTO_SPLITS)
    assert manifest == data.photo_source_manifest(photo_directory)
    sets = {split: {row["sha256"] for row in manifest if row["split"] == split}
            for split in data.PHOTO_SPLITS}
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"])
    Image.open(photo_directory / "000.png").save(photo_directory / "duplicate.png")
    assert data.photo_source_manifest(photo_directory) == manifest
    for split in data.PHOTO_SPLITS:
        loaded = data.load_photo_sources(photo_directory, photo_split=split)
        assert {data._photo_digest(image) for image in loaded} == sets[split]


def test_classic_source_copy_remains_reserved(photo_directory, monkeypatch):
    reserved = np.asarray(Image.open(photo_directory / "000.png"))
    monkeypatch.setattr(data, "load_skimage", lambda name: reserved if name == "camera" else None)
    manifest = data.photo_source_manifest(photo_directory)
    assert data._photo_digest(reserved) not in {row["sha256"] for row in manifest}


def test_rgb_copy_of_reserved_builtin_uses_identical_grayscale_conversion(tmp_path, monkeypatch):
    rgb = np.random.default_rng(3).integers(0, 256, (72, 72, 3), dtype=np.uint8)
    Image.fromarray(rgb).save(tmp_path / "classic_copy.png")
    reserved = data._to_gray_u8(rgb)
    monkeypatch.setattr(data, "load_skimage", lambda name: reserved if name == "astronaut" else None)
    assert data.photo_source_manifest(tmp_path) == []


def test_empty_split_raises_instead_of_reusing_training_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "load_skimage", lambda name: None)
    with pytest.raises(ValueError, match="No photo sources"):
        data.load_photo_sources(tmp_path, photo_split="val")
    with pytest.raises(ValueError, match="photo_split"):
        data.load_photo_sources(tmp_path, photo_split="invalid")


def test_mmap_pool_round_trip_and_source_changes_invalidate_cache(photo_directory, tmp_path):
    cache = tmp_path / "cache"
    images, cats = data.load_or_build_pool(12, seed=7, photo_dir=photo_directory,
                                          photo_split="train", mmap=True, cache_dir=cache)
    assert isinstance(images, np.memmap)
    assert isinstance(cats, np.memmap)
    expected, expected_cats = np.array(images), np.array(cats)
    del images, cats
    loaded, loaded_cats = data.load_or_build_pool(12, seed=7, photo_dir=photo_directory,
                                                 photo_split="train", mmap=True, cache_dir=cache)
    np.testing.assert_array_equal(loaded, expected)
    np.testing.assert_array_equal(loaded_cats, expected_cats)
    assert not loaded.flags.writeable
    previous_meta = set(cache.glob("*.json"))
    Image.fromarray(np.random.default_rng(99).integers(0, 256, (72, 72), dtype=np.uint8)).save(photo_directory / "new.png")
    data.load_or_build_pool(12, seed=7, photo_dir=photo_directory,
                            photo_split="train", mmap=True, cache_dir=cache)
    assert len(set(cache.glob("*.json")) - previous_meta) == 1


def test_fixed_quads_request_the_selected_photo_population(photo_directory):
    first = data.fixed_quads("val", 2, photo_directory, photo_split="val")
    repeated = data.fixed_quads("val", 2, photo_directory, photo_split="val")
    for kind in data.FIXED_KINDS:
        assert first[kind].shape == (2, 4, 128, 128)
        np.testing.assert_array_equal(first[kind], repeated[kind])


def test_download_extraction_rejects_sibling_directory_with_matching_prefix(tmp_path):
    destination = tmp_path / "photos"
    destination.mkdir()
    with zipfile.ZipFile(destination / "DIV2K_valid_HR.zip", "w") as archive:
        archive.writestr("../photos_other/escaped.png", b"invalid")
    with pytest.raises(RuntimeError, match="unsafe path"):
        data.fetch_div2k(destination)
    assert not (tmp_path / "photos_other" / "escaped.png").exists()
