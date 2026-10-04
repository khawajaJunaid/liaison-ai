"""End to end over HTTP: lease review, unit linkage, issue reporting, and the unified unit view."""


def upload_lease(client, path):
    with open(path, "rb") as fh:
        res = client.post("/api/leases", files={"file": (path.name, fh, "application/pdf")})
    assert res.status_code == 200, res.text
    return res.json()


def report_issue(client, unit_id, photo_names, samples, note="", reporter="Sara"):
    files = [("photos", (n, open(samples["photos"] / n, "rb"), "image/jpeg")) for n in photo_names]
    return client.post("/api/issues", data={"unit_id": unit_id, "reporter": reporter, "note": note}, files=files)


def unit_status(client, unit_id):
    return next(u for u in client.get("/api/units").json() if u["unit_id"] == unit_id)["status"]


def test_good_lease_flow_links_unit_and_updates_occupancy(client, samples):
    lease = upload_lease(client, samples["good"])
    assert lease["unit_id"] == "MC-B-1204"
    assert [r["outcome"] for r in lease["rule_results"]] == ["PASS"] * 7
    assert unit_status(client, "MC-B-1204") == "available"  # nothing changes until a human accepts

    res = client.post(f"/api/leases/{lease['id']}/accept")
    assert res.status_code == 200 and res.json()["status"] == "accepted"
    assert unit_status(client, "MC-B-1204") == "occupied"

    view = client.get("/api/units/MC-B-1204").json()
    assert view["lease"]["id"] == lease["id"]
    assert view["unit"]["lease_id"] == lease["id"]


def test_issue_appears_against_the_units_lease(client, samples):
    lease = upload_lease(client, samples["good"])
    client.post(f"/api/leases/{lease['id']}/accept")

    res = report_issue(client, "MC-B-1204", ["ac_unit_water_leak.jpg"], samples, note="dripping onto the floor")
    assert res.status_code == 200
    issue = res.json()
    assert issue["lease_id"] == lease["id"]
    wo = issue["work_order"]
    assert "Water leak" in wo["title"] and "Apartment 1204" in wo["title"]
    assert wo["priority"] == "high" and wo["status"] == "draft"
    assert "Split AC unit" in wo["equipment"]

    view = client.get("/api/units/MC-B-1204").json()
    assert view["lease"]["id"] == lease["id"]
    assert [i["id"] for i in view["issues"]] == [issue["id"]]  # lease and issue on one screen


def test_work_order_accept_with_edit_and_reject(client, samples):
    issue = report_issue(client, "MC-B-1204", ["water_heater_rust_old.jpg"], samples).json()
    res = client.patch(f"/api/issues/{issue['id']}/work-order",
                       json={"action": "accept", "title": "Replace water heater - 1204"})
    assert res.json()["work_order"]["status"] == "accepted"
    assert res.json()["work_order"]["title"] == "Replace water heater - 1204"

    second = report_issue(client, "MC-B-1204", ["kitchen_sink_new.jpg"], samples).json()
    res = client.patch(f"/api/issues/{second['id']}/work-order", json={"action": "reject"})
    assert res.json()["work_order"]["status"] == "rejected"


def test_unrecognised_photo_is_flagged_for_a_human_not_invented(client, samples):
    issue = report_issue(client, "MC-B-1204", ["unlabelled_photo.jpg"], samples).json()
    photo = issue["photos"][0]
    assert photo["condition"] == "unknown" and photo["needs_review"] and not photo["damages"]
    assert issue["work_order"]["title"].startswith("Inspect reported issue")
    assert "need a human check" in issue["work_order"]["description"]


def test_defective_lease_cannot_be_linked_to_an_occupied_unit(client, samples):
    lease = upload_lease(client, samples["defective"])
    assert lease["unit_id"] == "MC-B-1205"
    outcomes = {r["rule_id"]: r["outcome"] for r in lease["rule_results"]}
    assert set(outcomes.values()) == {"FAIL"}
    res = client.post(f"/api/leases/{lease['id']}/accept")
    assert res.status_code == 409 and "occupied" in res.json()["detail"]
    assert unit_status(client, "MC-B-1205") == "occupied"  # unchanged


def test_override_reruns_validation_and_keeps_flag_decisions(client, samples):
    lease = upload_lease(client, samples["defective"])
    lid = lease["id"]
    flag_ids = {f["id"] for f in lease["flags"]}
    assert "unit.unit_ref" in flag_ids

    # a reviewer dismisses one flag, then corrects the unit: rules and flags are re-derived
    client.patch(f"/api/leases/{lid}/flags/conflict.commencement_date", json={"action": "reject"})
    res = client.patch(f"/api/leases/{lid}/fields/unit_ref", json={"action": "override", "value": "MC-B-0902"})
    assert res.status_code == 200
    updated = res.json()
    assert updated["unit_id"] == "MC-B-0902"
    assert {r["rule_id"]: r["outcome"] for r in updated["rule_results"]}["R7"] == "PASS"
    assert "unit.unit_ref" not in {f["id"] for f in updated["flags"]}
    dismissed = next(f for f in updated["flags"] if f["id"] == "conflict.commencement_date")
    assert dismissed["decision"] == "rejected"
    field = updated["fields"]["unit_ref"]
    assert field["decision"] == "overridden" and field["original_value"] == "MC-B-1205"


def test_high_severity_flags_must_be_decided_before_accepting(client, samples):
    lease = upload_lease(client, samples["defective"])
    lid = lease["id"]
    client.patch(f"/api/leases/{lid}/fields/unit_ref", json={"action": "override", "value": "MC-A-0301"})
    res = client.post(f"/api/leases/{lid}/accept")
    assert res.status_code == 409 and "high-severity" in res.json()["detail"]

    current = client.get(f"/api/leases/{lid}").json()
    for f in current["flags"]:
        if f["severity"] == "high":
            client.patch(f"/api/leases/{lid}/flags/{f['id']}", json={"action": "accept"})
    assert client.post(f"/api/leases/{lid}/accept").status_code == 200
    assert unit_status(client, "MC-A-0301") == "occupied"


def test_rejecting_a_field_makes_rules_treat_it_as_missing(client, samples):
    lease = upload_lease(client, samples["good"])
    res = client.patch(f"/api/leases/{lease['id']}/fields/deposit_amount", json={"action": "reject"})
    r1 = next(r for r in res.json()["rule_results"] if r["rule_id"] == "R1")
    assert r1["outcome"] == "NOT_DETERMINABLE"
    assert any(f["id"] == "missing.deposit_amount" for f in res.json()["flags"])


def test_decided_lease_is_locked(client, samples):
    lease = upload_lease(client, samples["good"])
    client.post(f"/api/leases/{lease['id']}/accept")
    res = client.patch(f"/api/leases/{lease['id']}/fields/monthly_rent", json={"action": "override", "value": "1"})
    assert res.status_code == 409
    assert client.post(f"/api/leases/{lease['id']}/accept").status_code == 409


def test_every_decision_is_audited(client, samples):
    lease = upload_lease(client, samples["good"])
    client.patch(f"/api/leases/{lease['id']}/fields/monthly_rent", json={"action": "override", "value": "9600"})
    client.post(f"/api/leases/{lease['id']}/accept")
    actions = [a["action"] for a in client.get("/api/units/MC-B-1204").json()["audit"]]
    assert {"agent.run", "field.override", "lease.accept"} <= set(actions)
    override = next(a for a in client.get("/api/units/MC-B-1204").json()["audit"] if a["action"] == "field.override")
    assert override["detail"]["old"] == 9500.0 and override["detail"]["new"] == 9600.0


def test_scanned_lease_without_ocr_is_flagged_at_the_api(client, samples):
    lease = upload_lease(client, samples["scanned"])
    assert lease["unit_id"] is None and any(f["kind"] == "scan" for f in lease["flags"])
    assert client.post(f"/api/leases/{lease['id']}/accept").status_code == 409


def test_source_page_preview_renders(client, samples):
    lease = upload_lease(client, samples["good"])
    res = client.get(f"/api/leases/{lease['id']}/page/1.png", params={"field": "monthly_rent"})
    assert res.status_code == 200 and res.headers["content-type"] == "image/png"
    assert res.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert client.get(f"/api/leases/{lease['id']}/page/9.png").status_code == 404


def test_input_validation(client, samples, tmp_path):
    fake = tmp_path / "notes.pdf"
    fake.write_bytes(b"not really a pdf")
    with open(fake, "rb") as fh:
        assert client.post("/api/leases", files={"file": ("notes.pdf", fh, "application/pdf")}).status_code == 415

    lease = upload_lease(client, samples["good"])
    bad_date = client.patch(f"/api/leases/{lease['id']}/fields/expiry_date", json={"action": "override", "value": "soon"})
    assert bad_date.status_code == 422
    assert client.patch(f"/api/leases/{lease['id']}/fields/nope", json={"action": "accept"}).status_code == 404
    assert client.get("/api/leases/lease_missing").status_code == 404
    assert client.get("/api/units/MC-Z-0000").status_code == 404

    assert report_issue(client, "MC-Z-0000", ["wall_crack_stain.jpg"], samples).status_code == 404
    no_photos = client.post("/api/issues", data={"unit_id": "MC-B-1204"})
    assert no_photos.status_code == 422
    exe = client.post("/api/issues", data={"unit_id": "MC-B-1204"},
                      files=[("photos", ("evil.exe", b"MZ", "application/octet-stream"))])
    assert exe.status_code == 415


def test_photo_download_cannot_escape_its_folder(client, samples):
    issue = report_issue(client, "MC-B-1204", ["wall_crack_stain.jpg"], samples).json()
    name = issue["photos"][0]["filename"]
    assert client.get(f"/api/issues/{issue['id']}/photos/{name}").status_code == 200
    assert client.get(f"/api/issues/{issue['id']}/photos/..%2F..%2Fapp.db").status_code == 404


def test_index_page_is_served(client):
    res = client.get("/")
    assert res.status_code == 200 and "text/html" in res.headers["content-type"]
