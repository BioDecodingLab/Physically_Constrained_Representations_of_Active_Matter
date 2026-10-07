from ampub.cli import main


def test_cli(final_fixture, tmp_path):
    out = tmp_path / "publication"
    main(
        [
            "--input",
            str(final_fixture),
            "--out",
            str(out),
            "--no-videos",
            "--software-test",
        ]
    )
    assert (out / "PUBLICATION_MANIFEST.json").exists()
    assert (out / "figures/main/Figure_4_dynamics.svg").exists()
    assert (out / "tables/main/Table_1_protocol.csv").exists()
    assert not (out / "documents").exists()
    assert not (out / "manuscript_values.md").exists()


def test_cli_links_verified_reference_movie(final_fixture, tmp_path, monkeypatch):
    import ampub.cli as cli
    import ampub.visuals as visual
    import ampub.videos as video
    import json

    # Orchestration fixture only: real visual verification and encoding are
    # covered by test_visual_assets, without substituting any research image.
    monkeypatch.setattr(cli, "generate_all_figures", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "generate_tables", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "generate_videos", lambda *args: [])
    monkeypatch.setattr(
        visual,
        "generate_visuals",
        lambda *args: {
            "dataset_examples": True,
            "latent_PCA": True,
            "reconstruction_gallery": True,
        },
    )
    calls = []

    def reference(root, out):
        calls.append(root)
        return out / "reference.mp4"

    monkeypatch.setattr(video, "generate_reference_video", reference)
    export = tmp_path / "export"
    out = tmp_path / "linked"
    out.mkdir()
    (out / "SHA256SUMS.txt").write_text("previous-render checksum")
    cli.main(
        [
            "--input",
            str(final_fixture),
            "--out",
            str(out),
            "--visuals",
            str(export),
            "--software-test",
        ]
    )
    assert calls == [export]
    manifest = json.loads((out / "PUBLICATION_MANIFEST.json").read_text())
    assert manifest["generated"]["videos"] == 1
    assert "  SHA256SUMS.txt" not in (out / "SHA256SUMS.txt").read_text()
