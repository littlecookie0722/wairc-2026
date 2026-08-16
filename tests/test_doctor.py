from types import SimpleNamespace

import src.doctor as doctor


def test_doctor_reports_missing_torch_without_exposing_import_details(monkeypatch):
    def fake_import(name):
        if name == "torch":
            raise ModuleNotFoundError("private environment path")
        return SimpleNamespace(__version__="test-version")

    monkeypatch.setattr(doctor, "import_module", fake_import)

    report = doctor.collect_doctor_report()

    assert report["status"] == "error"
    assert report["checks"]["torch"] == {"status": "error", "errorType": "ModuleNotFoundError"}
    assert report["checks"]["cuda"]["available"] is None
    assert "private environment path" not in str(report)
