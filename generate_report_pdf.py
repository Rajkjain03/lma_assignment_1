import subprocess
import shutil

def compile_pdf():
    print("[PDF Build] Compiling report.tex with pdflatex...")
    res = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-jobname=REPORT", "report.tex"], capture_output=True, text=True)
    if res.returncode == 0:
        print("[PDF Build] Successfully generated REPORT.pdf")
        shutil.copyfile("REPORT.pdf", "report.pdf")
    else:
        print("[PDF Build] pdflatex error:\n", res.stderr or res.stdout)

if __name__ == "__main__":
    compile_pdf()
