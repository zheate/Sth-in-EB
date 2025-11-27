# Unit tests for local data storage module.

import json
import tempfile
from pathlib import Path
import uuid

import pandas as pd
import pytest

import sys
from pathlib import Path

# Add app directory to path for imports
app_dir = str(Path(__file__).parent.parent / "app")
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

from utils.local_storage import LocalDataStore, DataCategory, DatasetMetadata
from utils.exceptions import StorageError, DatasetNotFoundError, CorruptedDataError


class TestLocalDataStore:
    """Tests for LocalDataStore class"""

    @pytest.fixture
    def temp_store(self, tmp_path):
        """Create a LocalDataStore with a temporary directory"""
        return LocalDataStore(base_path=tmp_path)

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing"""
        return pd.DataFrame({
            "col_a": [1, 2, 3],
            "col_b": ["x", "y", "z"],
            "col_c": [1.1, 2.2, 3.3]
        })

    def test_init_creates_directories(self, tmp_path):
        """Test that initialization creates category directories"""
        store = LocalDataStore(base_path=tmp_path)
        
        for category in DataCategory:
            category_path = tmp_path / category.value
            assert category_path.exists()
            assert category_path.is_dir()

    def test_save_creates_parquet_file(self, temp_store, sample_df, tmp_path):
        """Test that save creates a parquet file"""
        dataset_id = temp_store.save(sample_df, DataCategory.EXTRACTION)
        
        parquet_path = tmp_path / "extraction" / f"{dataset_id}.parquet"
        assert parquet_path.exists()

    def test_save_creates_metadata_file(self, temp_store, sample_df, tmp_path):
        """Test that save creates a metadata JSON file"""
        dataset_id = temp_store.save(sample_df, DataCategory.EXTRACTION)
        
        meta_path = tmp_path / "extraction" / f"{dataset_id}.meta.json"
        assert meta_path.exists()
        
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        assert metadata["id"] == dataset_id
        assert metadata["category"] == "extraction"
        assert metadata["row_count"] == 3
        assert metadata["columns"] == ["col_a", "col_b", "col_c"]

    def test_save_returns_valid_uuid(self, temp_store, sample_df):
        """Test that save returns a valid UUID"""
        dataset_id = temp_store.save(sample_df, DataCategory.EXTRACTION)
        
        # Should not raise an exception
        uuid.UUID(dataset_id)

    def test_save_with_custom_filename(self, temp_store, sample_df, tmp_path):
        """Test save with custom filename"""
        dataset_id = temp_store.save(
            sample_df, 
            DataCategory.EXTRACTION,
            custom_filename="my_custom_file"
        )
        
        meta_path = tmp_path / "extraction" / f"{dataset_id}.meta.json"
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        assert "my_custom_file" in metadata["name"]

    def test_save_with_note(self, temp_store, sample_df, tmp_path):
        """Test save with user note"""
        dataset_id = temp_store.save(
            sample_df,
            DataCategory.PROGRESS,
            note="Test note"
        )
        
        meta_path = tmp_path / "progress" / f"{dataset_id}.meta.json"
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        assert metadata["note"] == "Test note"

    def test_save_with_extra_data(self, temp_store, sample_df, tmp_path):
        """Test save with extra data creates extra.json file"""
        extra = {"key": "value", "nested": {"a": 1}}
        dataset_id = temp_store.save(
            sample_df,
            DataCategory.ANALYSIS,
            extra_data=extra
        )
        
        extra_path = tmp_path / "analysis" / f"{dataset_id}.extra.json"
        assert extra_path.exists()
        
        with open(extra_path, 'r', encoding='utf-8') as f:
            loaded_extra = json.load(f)
        
        assert loaded_extra == extra

    def test_save_without_extra_data_no_extra_file(self, temp_store, sample_df, tmp_path):
        """Test that save without extra_data does not create extra.json"""
        dataset_id = temp_store.save(sample_df, DataCategory.EXTRACTION)
        
        extra_path = tmp_path / "extraction" / f"{dataset_id}.extra.json"
        assert not extra_path.exists()


class TestLoadMethod:
    """Tests for load() method"""

    @pytest.fixture
    def temp_store(self, tmp_path):
        """Create a LocalDataStore with a temporary directory"""
        return LocalDataStore(base_path=tmp_path)

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing"""
        return pd.DataFrame({
            "col_a": [1, 2, 3],
            "col_b": ["x", "y", "z"],
            "col_c": [1.1, 2.2, 3.3]
        })

    def test_load_returns_correct_dataframe(self, temp_store, sample_df):
        """Test that load returns the same DataFrame that was saved"""
        dataset_id = temp_store.save(sample_df, DataCategory.EXTRACTION)
        
        loaded_df, metadata, extra = temp_store.load(dataset_id)
        
        pd.testing.assert_frame_equal(loaded_df, sample_df)

    def test_load_returns_correct_metadata(self, temp_store, sample_df):
        """Test that load returns correct metadata"""
        dataset_id = temp_store.save(
            sample_df, 
            DataCategory.PROGRESS,
            name="Test Dataset",
            note="Test note"
        )
        
        loaded_df, metadata, extra = temp_store.load(dataset_id)
        
        assert metadata.id == dataset_id
        assert metadata.name == "Test Dataset"
        assert metadata.category == DataCategory.PROGRESS
        assert metadata.note == "Test note"
        assert metadata.row_count == 3
        assert metadata.columns == ["col_a", "col_b", "col_c"]

    def test_load_returns_extra_data(self, temp_store, sample_df):
        """Test that load returns extra data when present"""
        extra_data = {"key": "value", "nested": {"a": 1}}
        dataset_id = temp_store.save(
            sample_df,
            DataCategory.ANALYSIS,
            extra_data=extra_data
        )
        
        loaded_df, metadata, extra = temp_store.load(dataset_id)
        
        assert extra == extra_data

    def test_load_returns_none_extra_when_not_present(self, temp_store, sample_df):
        """Test that load returns None for extra when not saved"""
        dataset_id = temp_store.save(sample_df, DataCategory.EXTRACTION)
        
        loaded_df, metadata, extra = temp_store.load(dataset_id)
        
        assert extra is None

    def test_load_nonexistent_raises_not_found(self, temp_store):
        """Test that loading non-existent dataset raises DatasetNotFoundError"""
        fake_id = str(uuid.uuid4())
        
        with pytest.raises(DatasetNotFoundError) as exc_info:
            temp_store.load(fake_id)
        
        assert "未找到指定的数据集" in str(exc_info.value)

    def test_load_corrupted_metadata_raises_error(self, temp_store, sample_df, tmp_path):
        """Test that corrupted metadata file raises CorruptedDataError"""
        dataset_id = temp_store.save(sample_df, DataCategory.EXTRACTION)
        
        # Corrupt the metadata file
        meta_path = tmp_path / "extraction" / f"{dataset_id}.meta.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            f.write("not valid json {{{")
        
        with pytest.raises(CorruptedDataError) as exc_info:
            temp_store.load(dataset_id)
        
        assert "元数据文件损坏" in str(exc_info.value)

    def test_load_corrupted_parquet_raises_error(self, temp_store, sample_df, tmp_path):
        """Test that corrupted parquet file raises CorruptedDataError"""
        dataset_id = temp_store.save(sample_df, DataCategory.EXTRACTION)
        
        # Corrupt the parquet file
        parquet_path = tmp_path / "extraction" / f"{dataset_id}.parquet"
        with open(parquet_path, 'wb') as f:
            f.write(b"not valid parquet data")
        
        with pytest.raises(CorruptedDataError) as exc_info:
            temp_store.load(dataset_id)
        
        assert "数据文件损坏" in str(exc_info.value)

    def test_load_missing_parquet_raises_not_found(self, temp_store, sample_df, tmp_path):
        """Test that missing parquet file raises DatasetNotFoundError"""
        dataset_id = temp_store.save(sample_df, DataCategory.EXTRACTION)
        
        # Delete the parquet file but keep metadata
        parquet_path = tmp_path / "extraction" / f"{dataset_id}.parquet"
        parquet_path.unlink()
        
        with pytest.raises(DatasetNotFoundError) as exc_info:
            temp_store.load(dataset_id)
        
        assert "数据文件不存在" in str(exc_info.value)

    def test_load_corrupted_extra_raises_error(self, temp_store, sample_df, tmp_path):
        """Test that corrupted extra.json raises CorruptedDataError"""
        extra_data = {"key": "value"}
        dataset_id = temp_store.save(
            sample_df,
            DataCategory.EXTRACTION,
            extra_data=extra_data
        )
        
        # Corrupt the extra file
        extra_path = tmp_path / "extraction" / f"{dataset_id}.extra.json"
        with open(extra_path, 'w', encoding='utf-8') as f:
            f.write("invalid json [[[")
        
        with pytest.raises(CorruptedDataError) as exc_info:
            temp_store.load(dataset_id)
        
        assert "扩展数据文件损坏" in str(exc_info.value)

    def test_load_from_different_categories(self, temp_store, sample_df):
        """Test that load works for datasets in different categories"""
        # Save to different categories
        id_extraction = temp_store.save(sample_df, DataCategory.EXTRACTION)
        id_progress = temp_store.save(sample_df, DataCategory.PROGRESS)
        id_analysis = temp_store.save(sample_df, DataCategory.ANALYSIS)
        
        # Load from each category
        df1, meta1, _ = temp_store.load(id_extraction)
        df2, meta2, _ = temp_store.load(id_progress)
        df3, meta3, _ = temp_store.load(id_analysis)
        
        assert meta1.category == DataCategory.EXTRACTION
        assert meta2.category == DataCategory.PROGRESS
        assert meta3.category == DataCategory.ANALYSIS


class TestSanitizeFilename:
    """Tests for filename sanitization"""

    @pytest.fixture
    def store(self, tmp_path):
        return LocalDataStore(base_path=tmp_path)

    def test_sanitize_removes_unsafe_chars(self, store):
        """Test that unsafe characters are replaced"""
        result = store._sanitize_filename('file<>:"/\\|?*name')
        assert '<' not in result
        assert '>' not in result
        assert ':' not in result
        assert '"' not in result
        assert '/' not in result
        assert '\\' not in result
        assert '|' not in result
        assert '?' not in result
        assert '*' not in result

    def test_sanitize_collapses_whitespace(self, store):
        """Test that multiple spaces/underscores are collapsed"""
        result = store._sanitize_filename('file   name__test')
        assert '   ' not in result
        assert '__' not in result

    def test_sanitize_empty_returns_unnamed(self, store):
        """Test that empty string returns 'unnamed'"""
        result = store._sanitize_filename('')
        assert result == "unnamed"

    def test_sanitize_only_unsafe_chars_returns_unnamed(self, store):
        """Test that string with only unsafe chars returns 'unnamed'"""
        result = store._sanitize_filename('<>:"/\\|?*')
        assert result == "unnamed"

    def test_sanitize_preserves_chinese(self, store):
        """Test that Chinese characters are preserved"""
        result = store._sanitize_filename('测试文件名')
        assert '测试文件名' in result

    def test_sanitize_limits_length(self, store):
        """Test that filename length is limited"""
        long_name = 'a' * 300
        result = store._sanitize_filename(long_name)
        assert len(result) <= 200


class TestGenerateFilename:
    """Tests for filename generation"""

    @pytest.fixture
    def store(self, tmp_path):
        return LocalDataStore(base_path=tmp_path)

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    def test_auto_generate_contains_category(self, store, sample_df):
        """Test auto-generated filename contains category"""
        result = store._generate_filename(sample_df, DataCategory.EXTRACTION)
        assert "extraction" in result

    def test_auto_generate_contains_row_count(self, store, sample_df):
        """Test auto-generated filename contains row count"""
        result = store._generate_filename(sample_df, DataCategory.EXTRACTION)
        assert "3rows" in result

    def test_auto_generate_contains_col_count(self, store, sample_df):
        """Test auto-generated filename contains column count"""
        result = store._generate_filename(sample_df, DataCategory.EXTRACTION)
        assert "2cols" in result

    def test_custom_filename_used(self, store, sample_df):
        """Test that custom filename is used when provided"""
        result = store._generate_filename(
            sample_df, 
            DataCategory.EXTRACTION,
            custom_filename="my_file"
        )
        assert result == "my_file"

    def test_custom_filename_sanitized(self, store, sample_df):
        """Test that custom filename is sanitized"""
        result = store._generate_filename(
            sample_df,
            DataCategory.EXTRACTION,
            custom_filename="my<>file"
        )
        assert '<' not in result
        assert '>' not in result


class TestDeleteMethod:
    """Tests for delete() method"""

    @pytest.fixture
    def temp_store(self, tmp_path):
        """Create a LocalDataStore with a temporary directory"""
        return LocalDataStore(base_path=tmp_path)

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing"""
        return pd.DataFrame({
            "col_a": [1, 2, 3],
            "col_b": ["x", "y", "z"],
        })

    def test_delete_removes_parquet_file(self, temp_store, sample_df, tmp_path):
        """Test that delete removes the parquet file"""
        dataset_id = temp_store.save(sample_df, DataCategory.EXTRACTION)
        parquet_path = tmp_path / "extraction" / f"{dataset_id}.parquet"
        
        assert parquet_path.exists()
        
        result = temp_store.delete(dataset_id)
        
        assert result is True
        assert not parquet_path.exists()

    def test_delete_removes_metadata_file(self, temp_store, sample_df, tmp_path):
        """Test that delete removes the metadata file"""
        dataset_id = temp_store.save(sample_df, DataCategory.EXTRACTION)
        meta_path = tmp_path / "extraction" / f"{dataset_id}.meta.json"
        
        assert meta_path.exists()
        
        temp_store.delete(dataset_id)
        
        assert not meta_path.exists()

    def test_delete_removes_extra_file(self, temp_store, sample_df, tmp_path):
        """Test that delete removes the extra.json file when present"""
        extra_data = {"key": "value"}
        dataset_id = temp_store.save(
            sample_df, 
            DataCategory.EXTRACTION,
            extra_data=extra_data
        )
        extra_path = tmp_path / "extraction" / f"{dataset_id}.extra.json"
        
        assert extra_path.exists()
        
        temp_store.delete(dataset_id)
        
        assert not extra_path.exists()

    def test_delete_removes_all_associated_files(self, temp_store, sample_df, tmp_path):
        """Test that delete removes all associated files at once"""
        extra_data = {"key": "value"}
        dataset_id = temp_store.save(
            sample_df,
            DataCategory.ANALYSIS,
            extra_data=extra_data
        )
        
        parquet_path = tmp_path / "analysis" / f"{dataset_id}.parquet"
        meta_path = tmp_path / "analysis" / f"{dataset_id}.meta.json"
        extra_path = tmp_path / "analysis" / f"{dataset_id}.extra.json"
        
        # All files should exist before delete
        assert parquet_path.exists()
        assert meta_path.exists()
        assert extra_path.exists()
        
        result = temp_store.delete(dataset_id)
        
        # All files should be removed after delete
        assert result is True
        assert not parquet_path.exists()
        assert not meta_path.exists()
        assert not extra_path.exists()

    def test_delete_nonexistent_raises_not_found(self, temp_store):
        """Test that deleting non-existent dataset raises DatasetNotFoundError"""
        fake_id = str(uuid.uuid4())
        
        with pytest.raises(DatasetNotFoundError) as exc_info:
            temp_store.delete(fake_id)
        
        assert "未找到指定的数据集" in str(exc_info.value)

    def test_delete_returns_true_on_success(self, temp_store, sample_df):
        """Test that delete returns True on successful deletion"""
        dataset_id = temp_store.save(sample_df, DataCategory.PROGRESS)
        
        result = temp_store.delete(dataset_id)
        
        assert result is True

    def test_delete_dataset_no_longer_loadable(self, temp_store, sample_df):
        """Test that deleted dataset cannot be loaded"""
        dataset_id = temp_store.save(sample_df, DataCategory.EXTRACTION)
        
        # Should be loadable before delete
        df, meta, extra = temp_store.load(dataset_id)
        assert df is not None
        
        # Delete the dataset
        temp_store.delete(dataset_id)
        
        # Should raise error after delete
        with pytest.raises(DatasetNotFoundError):
            temp_store.load(dataset_id)

    def test_delete_dataset_no_longer_in_list(self, temp_store, sample_df):
        """Test that deleted dataset no longer appears in list"""
        dataset_id = temp_store.save(sample_df, DataCategory.EXTRACTION)
        
        # Should be in list before delete
        datasets = temp_store.list_datasets()
        assert any(d.id == dataset_id for d in datasets)
        
        # Delete the dataset
        temp_store.delete(dataset_id)
        
        # Should not be in list after delete
        datasets = temp_store.list_datasets()
        assert not any(d.id == dataset_id for d in datasets)

    def test_delete_works_for_all_categories(self, temp_store, sample_df, tmp_path):
        """Test that delete works for datasets in all categories"""
        for category in DataCategory:
            dataset_id = temp_store.save(sample_df, category)
            parquet_path = tmp_path / category.value / f"{dataset_id}.parquet"
            
            assert parquet_path.exists()
            
            result = temp_store.delete(dataset_id)
            
            assert result is True
            assert not parquet_path.exists()


class TestExportToExcel:
    """Tests for export_to_excel() method"""

    @pytest.fixture
    def temp_store(self, tmp_path):
        """Create a LocalDataStore with a temporary directory"""
        return LocalDataStore(base_path=tmp_path)

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing"""
        return pd.DataFrame({
            "col_a": [1, 2, 3],
            "col_b": ["x", "y", "z"],
            "col_c": [1.1, 2.2, 3.3]
        })

    def test_export_single_dataset_creates_file(self, temp_store, sample_df, tmp_path):
        """Test that export creates an Excel file"""
        dataset_id = temp_store.save(sample_df, DataCategory.EXTRACTION)
        output_path = tmp_path / "test_export.xlsx"
        
        result = temp_store.export_to_excel([dataset_id], output_path)
        
        assert result.exists()
        assert result == output_path

    def test_export_auto_generates_filename(self, temp_store, sample_df, tmp_path):
        """Test that export auto-generates filename when not provided"""
        dataset_id = temp_store.save(sample_df, DataCategory.EXTRACTION)
        
        result = temp_store.export_to_excel([dataset_id])
        
        assert result.exists()
        assert "extraction" in result.name
        assert "export" in result.name
        assert ".xlsx" in result.name

    def test_export_filename_contains_date(self, temp_store, sample_df):
        """Test that auto-generated filename contains date"""
        from datetime import datetime
        dataset_id = temp_store.save(sample_df, DataCategory.EXTRACTION)
        
        result = temp_store.export_to_excel([dataset_id])
        
        date_str = datetime.now().strftime("%Y%m%d")
        assert date_str in result.name

    def test_export_with_summary_creates_two_sheets(self, temp_store, sample_df, tmp_path):
        """Test that export with summary creates data and summary sheets"""
        dataset_id = temp_store.save(sample_df, DataCategory.EXTRACTION)
        output_path = tmp_path / "test_export.xlsx"
        
        temp_store.export_to_excel([dataset_id], output_path, include_summary=True)
        
        # Read the Excel file and check sheets
        excel_file = pd.ExcelFile(output_path)
        sheet_names = excel_file.sheet_names
        
        assert '数据' in sheet_names
        assert '统计摘要' in sheet_names
        assert len(sheet_names) >= 2

    def test_export_without_summary_creates_one_sheet(self, temp_store, sample_df, tmp_path):
        """Test that export without summary creates only data sheet"""
        dataset_id = temp_store.save(sample_df, DataCategory.EXTRACTION)
        output_path = tmp_path / "test_export.xlsx"
        
        temp_store.export_to_excel([dataset_id], output_path, include_summary=False)
        
        excel_file = pd.ExcelFile(output_path)
        sheet_names = excel_file.sheet_names
        
        assert '数据' in sheet_names
        assert '统计摘要' not in sheet_names

    def test_export_data_sheet_contains_correct_data(self, temp_store, sample_df, tmp_path):
        """Test that exported data sheet contains correct data"""
        dataset_id = temp_store.save(sample_df, DataCategory.EXTRACTION)
        output_path = tmp_path / "test_export.xlsx"
        
        temp_store.export_to_excel([dataset_id], output_path)
        
        loaded_df = pd.read_excel(output_path, sheet_name='数据')
        pd.testing.assert_frame_equal(loaded_df, sample_df)

    def test_export_batch_datasets(self, temp_store, sample_df, tmp_path):
        """Test batch export of multiple datasets"""
        df1 = sample_df.copy()
        df2 = pd.DataFrame({"col_a": [4, 5], "col_b": ["a", "b"], "col_c": [4.4, 5.5]})
        
        id1 = temp_store.save(df1, DataCategory.EXTRACTION, name="Dataset1")
        id2 = temp_store.save(df2, DataCategory.EXTRACTION, name="Dataset2")
        
        output_path = tmp_path / "batch_export.xlsx"
        result = temp_store.export_to_excel([id1, id2], output_path)
        
        assert result.exists()
        
        # Check merged data
        loaded_df = pd.read_excel(output_path, sheet_name='数据')
        # Should have 5 rows total (3 + 2)
        assert len(loaded_df) == 5
        # Should have dataset identifier columns
        assert '_数据集' in loaded_df.columns
        assert '_数据集ID' in loaded_df.columns

    def test_export_batch_contains_all_datasets(self, temp_store, sample_df, tmp_path):
        """Test that batch export contains data from all datasets"""
        df1 = pd.DataFrame({"value": [1, 2, 3]})
        df2 = pd.DataFrame({"value": [4, 5, 6]})
        df3 = pd.DataFrame({"value": [7, 8, 9]})
        
        id1 = temp_store.save(df1, DataCategory.EXTRACTION, name="D1")
        id2 = temp_store.save(df2, DataCategory.EXTRACTION, name="D2")
        id3 = temp_store.save(df3, DataCategory.EXTRACTION, name="D3")
        
        output_path = tmp_path / "batch_export.xlsx"
        temp_store.export_to_excel([id1, id2, id3], output_path)
        
        loaded_df = pd.read_excel(output_path, sheet_name='数据')
        
        # Should have 9 rows total
        assert len(loaded_df) == 9
        # Should contain all values
        all_values = loaded_df['value'].tolist()
        for v in range(1, 10):
            assert v in all_values

    def test_export_nonexistent_dataset_raises_error(self, temp_store):
        """Test that exporting non-existent dataset raises error"""
        fake_id = str(uuid.uuid4())
        
        with pytest.raises(DatasetNotFoundError):
            temp_store.export_to_excel([fake_id])

    def test_export_empty_list_raises_error(self, temp_store):
        """Test that exporting empty list raises ExportError"""
        from utils.exceptions import ExportError
        
        with pytest.raises(ExportError) as exc_info:
            temp_store.export_to_excel([])
        
        assert "未指定要导出的数据集" in str(exc_info.value)

    def test_export_mixed_categories_uses_mixed_filename(self, temp_store, tmp_path):
        """Test that mixed category export uses 'mixed' in filename"""
        df = pd.DataFrame({"a": [1, 2]})
        
        id1 = temp_store.save(df, DataCategory.EXTRACTION)
        id2 = temp_store.save(df, DataCategory.PROGRESS)
        
        result = temp_store.export_to_excel([id1, id2])
        
        assert "mixed" in result.name


class TestExportToCsv:
    """Tests for export_to_csv() method"""

    @pytest.fixture
    def temp_store(self, tmp_path):
        """Create a LocalDataStore with a temporary directory"""
        return LocalDataStore(base_path=tmp_path)

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing"""
        return pd.DataFrame({
            "col_a": [1, 2, 3],
            "col_b": ["x", "y", "z"],
            "col_c": [1.1, 2.2, 3.3]
        })

    def test_export_csv_creates_file(self, temp_store, sample_df, tmp_path):
        """Test that export creates a CSV file"""
        dataset_id = temp_store.save(sample_df, DataCategory.EXTRACTION)
        output_path = tmp_path / "test_export.csv"
        
        result = temp_store.export_to_csv(dataset_id, output_path)
        
        assert result.exists()
        assert result == output_path

    def test_export_csv_auto_generates_filename(self, temp_store, sample_df):
        """Test that export auto-generates filename when not provided"""
        dataset_id = temp_store.save(sample_df, DataCategory.PROGRESS)
        
        result = temp_store.export_to_csv(dataset_id)
        
        assert result.exists()
        assert "progress" in result.name
        assert "export" in result.name
        assert ".csv" in result.name

    def test_export_csv_filename_contains_date(self, temp_store, sample_df):
        """Test that auto-generated CSV filename contains date"""
        from datetime import datetime
        dataset_id = temp_store.save(sample_df, DataCategory.ANALYSIS)
        
        result = temp_store.export_to_csv(dataset_id)
        
        date_str = datetime.now().strftime("%Y%m%d")
        assert date_str in result.name

    def test_export_csv_contains_correct_data(self, temp_store, sample_df, tmp_path):
        """Test that exported CSV contains correct data"""
        dataset_id = temp_store.save(sample_df, DataCategory.EXTRACTION)
        output_path = tmp_path / "test_export.csv"
        
        temp_store.export_to_csv(dataset_id, output_path)
        
        loaded_df = pd.read_csv(output_path)
        pd.testing.assert_frame_equal(loaded_df, sample_df)

    def test_export_csv_uses_utf8_bom(self, temp_store, sample_df, tmp_path):
        """Test that CSV uses UTF-8-BOM encoding"""
        # Create DataFrame with Chinese characters
        df_chinese = pd.DataFrame({
            "名称": ["测试1", "测试2"],
            "数值": [1, 2]
        })
        dataset_id = temp_store.save(df_chinese, DataCategory.EXTRACTION)
        output_path = tmp_path / "test_chinese.csv"
        
        temp_store.export_to_csv(dataset_id, output_path)
        
        # Read raw bytes to check BOM
        with open(output_path, 'rb') as f:
            raw_bytes = f.read(3)
        
        # UTF-8 BOM is EF BB BF
        assert raw_bytes == b'\xef\xbb\xbf'

    def test_export_csv_chinese_readable(self, temp_store, tmp_path):
        """Test that exported CSV with Chinese is readable"""
        df_chinese = pd.DataFrame({
            "名称": ["测试数据", "中文内容"],
            "数值": [100, 200]
        })
        dataset_id = temp_store.save(df_chinese, DataCategory.EXTRACTION)
        output_path = tmp_path / "test_chinese.csv"
        
        temp_store.export_to_csv(dataset_id, output_path)
        
        # Read back and verify
        loaded_df = pd.read_csv(output_path, encoding='utf-8-sig')
        pd.testing.assert_frame_equal(loaded_df, df_chinese)

    def test_export_csv_nonexistent_dataset_raises_error(self, temp_store):
        """Test that exporting non-existent dataset raises error"""
        fake_id = str(uuid.uuid4())
        
        with pytest.raises(DatasetNotFoundError):
            temp_store.export_to_csv(fake_id)

    def test_export_csv_filename_contains_category(self, temp_store, sample_df):
        """Test that auto-generated filename contains category"""
        for category in DataCategory:
            dataset_id = temp_store.save(sample_df, category)
            result = temp_store.export_to_csv(dataset_id)
            assert category.value in result.name
