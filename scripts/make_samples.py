"""Generate sample leases and photos. The brief did not ship any, so these stand in.

    python -m scripts.make_samples            # writes into ./samples

lease_good.pdf        clean lease for MC-B-1204 (available): should pass every rule
lease_defective.pdf   seeded problems: occupied unit, low deposit, 48-month term that
                      disagrees with its dates, conflicting start dates, vague escalation,
                      annual rent that does not reconcile, unsigned tenant
lease_scanned.pdf     the good lease rasterised, so it has no text layer (OCR path)
photos/*.jpg          placeholder images; the stub vision model reads the filenames
"""
from __future__ import annotations

import sys
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

GOOD = [
    ("h", "LEASE AGREEMENT"),
    ("t", "Marina Crest Residences, Lusail Marina District, Doha"),
    ("h", "1. PARTIES"),
    ("t", "Landlord: Marina Crest Holdings W.L.L."),
    ("t", "Tenant: Sara Al-Mansoori"),
    ("h", "2. PREMISES"),
    ("t", "Unit: Apartment 1204, Tower B (MC-B-1204), Parking Bay B-77"),
    ("h", "3. TERM"),
    ("t", "Commencement Date: 1 March 2026"),
    ("t", "Expiry Date: 28 February 2027"),
    ("t", "This lease is for a fixed term of twelve (12) months."),
    ("h", "4. RENT AND DEPOSIT"),
    ("t", "Monthly Rent: QAR 9,500, payable monthly in advance."),
    ("t", "Annual Rent: QAR 114,000."),
    ("t", "Security Deposit: QAR 9,500, held for the term of the lease."),
    ("h", "5. RENT ESCALATION"),
    ("t", "At each anniversary of the Commencement Date the Monthly Rent increases"),
    ("t", "by 5% over the rent of the preceding year."),
    ("h", "6. RENEWAL"),
    ("t", "The Tenant may renew for a further term by written request given not"),
    ("t", "less than 90 days before expiry, subject to the escalation above."),
    ("h", "7. TERMINATION"),
    ("t", "Either party may terminate early on 60 days written notice; the Tenant"),
    ("t", "forfeits one month of rent if terminating before month six."),
    ("h", "8. SIGNATURES"),
    ("t", "Landlord Signature: /s/ Omar Al-Thani        Date: 20 February 2026"),
    ("t", "Tenant Signature: /s/ Sara Al-Mansoori        Date: 20 February 2026"),
]

DEFECTIVE = [
    ("h", "LEASE AGREEMENT"),
    ("t", "Marina Crest Residences, Lusail Marina District, Doha"),
    ("t", "This lease commences on 15 April 2026 between the parties named below."),
    ("h", "1. PARTIES"),
    ("t", "Landlord: Marina Crest Holdings W.L.L."),
    ("t", "Tenant: Khalid Nasser"),
    ("h", "2. PREMISES"),
    ("t", "Unit: Apartment 1205, Tower B (MC-B-1205)"),
    ("h", "3. TERM"),
    ("t", "Commencement Date: 1 April 2026"),
    ("t", "Expiry Date: 31 March 2029"),
    ("t", "This lease is for a fixed term of forty-eight (48) months."),
    ("h", "4. RENT AND DEPOSIT"),
    ("t", "Monthly Rent: QAR 12,000, payable monthly in advance."),
    ("t", "Annual Rent: QAR 140,000."),
    ("t", "Security Deposit: QAR 6,000."),
    ("h", "5. RENT ESCALATION"),
    ("t", "Rent shall be adjusted from time to time as mutually agreed by the parties."),
    ("h", "6. TERMINATION"),
    ("t", "Either party may terminate early on 30 days written notice."),
    ("h", "7. SIGNATURES"),
    ("t", "Landlord Signature: /s/ Omar Al-Thani        Date: 25 March 2026"),
    ("t", "Tenant Signature: ____________________        Date: ____________"),
]


def _lease_pdf(path: Path, rows: list[tuple[str, str]]) -> None:
    c = canvas.Canvas(str(path), pagesize=A4)
    y = 790
    for kind, text in rows:
        c.setFont("Helvetica-Bold" if kind == "h" else "Helvetica", 12 if kind == "h" else 10.5)
        y -= 10 if kind == "h" else 0
        c.drawString(56, y, text)
        y -= 18
    c.save()


def _scanned_pdf(source: Path, path: Path) -> None:
    page = pymupdf.open(source)[0]
    png = page.get_pixmap(matrix=pymupdf.Matrix(2, 2)).tobytes("png")
    out = pymupdf.open()
    new = out.new_page(width=page.rect.width, height=page.rect.height)
    new.insert_image(new.rect, stream=png)
    out.save(path)


PHOTOS = {
    "ac_unit_water_leak.jpg": (70, 110, 150),
    "water_heater_rust_old.jpg": (140, 90, 60),
    "kitchen_sink_new.jpg": (190, 190, 200),
    "wall_crack_stain.jpg": (200, 190, 160),
    "unlabelled_photo.jpg": (120, 120, 120),
}


def _photos(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name, colour in PHOTOS.items():
        img = Image.new("RGB", (640, 420), colour)
        ImageDraw.Draw(img).text((20, 20), f"placeholder: {name}", fill=(255, 255, 255))
        img.save(folder / name, quality=80)


def build_all(out: Path) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = {"good": out / "lease_good.pdf", "defective": out / "lease_defective.pdf",
             "scanned": out / "lease_scanned.pdf", "photos": out / "photos"}
    _lease_pdf(paths["good"], GOOD)
    _lease_pdf(paths["defective"], DEFECTIVE)
    _scanned_pdf(paths["good"], paths["scanned"])
    _photos(paths["photos"])
    return paths


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "samples"
    for name, p in build_all(target).items():
        print(f"{name:10s} {p}")
