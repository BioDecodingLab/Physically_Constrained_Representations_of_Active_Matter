from pathlib import Path
from ampub.schema import load_final
from ampub.figures import generate_all_figures
from ampub.tables import generate_tables
from ampub.videos import generate_videos
from ampub.results import key_results


def test_figures_tables_values(final_fixture, tmp_path):
    data = load_final(final_fixture)
    generate_all_figures(data, tmp_path / "figures")
    generate_tables(data, tmp_path / "tables")
    assert len(list((tmp_path / "figures").rglob("*.svg"))) >= 7
    assert len(list((tmp_path / "figures").rglob("*.png"))) >= 7
    assert (
        tmp_path / "tables/main/Table_2_reconstruction_key_results.csv"
    ).stat().st_size > 20
    k = key_results(data)
    assert (k["Q_reduction_percent"] > 0).all()


def test_videos(final_fixture, tmp_path):
    data = load_final(final_fixture)
    videos = generate_videos(data, tmp_path / "videos")
    assert len(videos) == 2
    assert all(v.exists() and v.stat().st_size > 1000 for v in videos)


def test_table_legends_and_asset_numbering(final_fixture, tmp_path):
    import pandas as pd
    from ampub.tables import TABLE_LEGENDS

    data = load_final(final_fixture)
    out = tmp_path / "tables"
    generate_tables(data, out)
    for name in TABLE_LEGENDS:
        if name.endswith("Table_S2_reconstruction_percentage_contrasts") and data.group_metrics.empty:
            assert not (out / name).with_suffix(".caption.md").exists()
            continue
        assert (out / name).with_suffix(".csv").exists()
        assert (out / name).with_suffix(".caption.md").read_text().strip()
    caption = (out / "supplementary/Table_S1_reconstruction_absolute_contrasts.caption.md").read_text()
    assert "difference_b_minus_a is AE minus R-PC-AE" in caption
    mapping = pd.read_csv(tmp_path / "FIGURE_DATA_MAP.csv").set_index("Display item")
    assert mapping.loc["Supplementary Video 1", "Input"] == "visuals.npz reference_movie"
    assert "77-lead" in mapping.loc["Supplementary Video 2", "Reporting"]
    assert "10-lead" in mapping.loc["Supplementary Video 3", "Reporting"]
