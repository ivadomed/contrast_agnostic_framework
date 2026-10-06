# old (val100 rung 5, from the .bak copies) vs new (val000) fill-swap rows, same code path
import sys, os, io, contextlib
sys.path.insert(0, "paper/scripts")
import compute_dissociation_pvalues as C
BAK = ".bak_20261006_pre_val000_rung5"
orig = list(C.ROWS)
for tag, rows in (("NEW (val000 where retrained)", orig),
                  ("OLD (val100)", [(l, b, r + BAK if os.path.exists(r + BAK) else r) for l, b, r in orig])):
    C.ROWS[:] = rows
    for metric in ("dice",):
        buf = io.StringIO()
        sys.argv = ["x", "--metric", metric]
        with contextlib.redirect_stdout(buf):
            C.main()
        print(f"######## {tag} {metric}\n" + buf.getvalue().split("--- LaTeX")[0])
