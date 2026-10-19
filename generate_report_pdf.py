import subprocess
import os

def compile_pdf():
    print("[PDF Build] Compiling report.tex with pdflatex...")
    res = subprocess.run(["pdflatex", "-interaction=nonstopmode", "report.tex"], capture_output=True, text=True)
    if res.returncode == 0:
        print("[PDF Build] Successfully generated report.pdf")
        for ext in [".aux", ".log", ".out"]:
            aux_file = f"report{ext}"
            if os.path.exists(aux_file):
                os.remove(aux_file)
    else:
        print("[PDF Build] pdflatex error:\n", res.stderr or res.stdout)

if __name__ == "__main__":
    compile_pdf()
