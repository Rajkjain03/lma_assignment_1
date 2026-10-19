import subprocess
import os

def compile_pdf():
    print("[PDF Build] Compiling report.tex with pdflatex (Pass 1)...")
    res1 = subprocess.run(["pdflatex", "-interaction=nonstopmode", "report.tex"], capture_output=True, text=True)
    if res1.returncode != 0:
        print("[PDF Build] pdflatex error in Pass 1:\n", res1.stderr or res1.stdout)
        return
    print("[PDF Build] Compiling report.tex with pdflatex (Pass 2 for cross-references)...")
    res2 = subprocess.run(["pdflatex", "-interaction=nonstopmode", "report.tex"], capture_output=True, text=True)
    if res2.returncode == 0:
        print("[PDF Build] Successfully generated report.pdf with all references resolved.")
        for ext in [".aux", ".log", ".out"]:
            aux_file = f"report{ext}"
            if os.path.exists(aux_file):
                os.remove(aux_file)
    else:
        print("[PDF Build] pdflatex error in Pass 2:\n", res2.stderr or res2.stdout)

if __name__ == "__main__":
    compile_pdf()
