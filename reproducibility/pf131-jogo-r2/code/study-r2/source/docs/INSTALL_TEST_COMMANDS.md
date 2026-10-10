# Installer et tester PF C++ 1.3.1 — METIS+LDL / METIS+MUMPS

Télécharger `pf_cpp_v1.3.1_source.tar.gz` dans `/data/projects/PF_GlobalOptimization`.
Cette nouvelle version nécessite une compilation. La bibliothèque HSL n'est
pas nécessaire pour les deux combinaisons METIS ci-dessous.

## Installation dans l'environnement existant

```bash
(
set -Eeo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pf_cpp_110
conda install --yes --freeze-installed --channel conda-forge metis mumps-seq mumps-include cmake ninja pkg-config
BASE=/data/projects/PF_GlobalOptimization
SRC="$BASE/pf_cpp_v1.3.1"
BUILD="$SRC/build_release"
test -s "$BASE/pf_cpp_v1.3.1_source.tar.gz"
test ! -e "$SRC"
tar -xzf "$BASE/pf_cpp_v1.3.1_source.tar.gz" -C "$BASE"
(cd "$SRC" && sha256sum --quiet -c MANIFEST.sha256)
unset LD_PRELOAD ASAN_OPTIONS UBSAN_OPTIONS LSAN_OPTIONS
export OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
cmake -S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" -DPython3_EXECUTABLE="$CONDA_PREFIX/bin/python" -DPF_NATIVE=ON -DPF_WITH_OPENMP=ON -DPF_WITH_IPOPT=ON -DPF_WITH_MUMPS=ON -DPF_WITH_METIS=ON -DPF_MUMPS_INCLUDE="$CONDA_PREFIX/include" -DPF_MUMPS_LIBRARY="$CONDA_PREFIX/lib/libdmumps_seq.so" -DPF_METIS_INCLUDE="$CONDA_PREFIX/include" -DPF_METIS_LIBRARY="$CONDA_PREFIX/lib/libmetis.so" -DBUILD_TESTING=ON
cmake --build "$BUILD" --parallel 8
ctest --test-dir "$BUILD" --output-on-failure
"$BUILD/pf_solver" --version
"$BUILD/pf_local_incumbent_tests"
"$BUILD/pf_linear_check" --backend ldl --ordering metis
"$BUILD/pf_linear_check" --backend mumps --ordering metis
"$BUILD/pf_linear_check" --backend ipopt-mumps
)
```

`--freeze-installed` demande de conserver les dépendances déjà installées.
Les compilateurs, Eigen, Ipopt et les dépendances Python proviennent de
l'environnement PF existant. Les commandes pour créer un environnement complet
figurent plus bas. Si le dossier 1.3.1 existe déjà, reprendre la compilation
sans extraction ; cette procédure ne l'écrase pas.

## LJ38 et LJ76 : deux variantes, entrées figées et témoin Ipopt

Les deux variantes raffinent toutes les particules. Le témoin Ipopt multistart
est lancé une seule fois par taille, dans le dossier `ldl`. Il utilise MUMPS
avec l'ordonnancement interne d'Ipopt ; METIS ci-dessous configure le Newton
natif. Les trois méthodes reçoivent exactement la même population initiale.
Le budget temps s'applique séparément à chaque méthode ; ce n'est pas un budget
global de toute la campagne. Le seuil de stagnation est désactivé explicitement.

```bash
(
set -Eeo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pf_cpp_110
BASE=/data/projects/PF_GlobalOptimization
BUILD="$BASE/pf_cpp_v1.3.1/build_release"
REFERENCE="$BASE/results_pf130_20260916_193554"
ATOMS=(38 76)
BACKENDS=(ldl mumps)
WORKERS=10
MAX_ITER=1000
MAX_TIME=3600
OUT=$(mktemp -d "$BASE/PF131_METIS_XXXXXXXX")
export OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
printf 'RESULTATS=%s\n' "$OUT"
for N in "${ATOMS[@]}"; do
  POP="$REFERENCE/LJ${N}_run1/initial.pfpop"
  test -s "$POP"
  for LINEAR in "${BACKENDS[@]}"; do
    EXTRA=()
    if [ "$LINEAR" = "${BACKENDS[0]}" ]; then EXTRA=(--include-ipopt); fi
    "$BUILD/pf_solver" --atoms "$N" --runs 1 --seed 2024 --population "$POP" --max-iter "$MAX_ITER" --no-improve-stop 0 --max-time "$MAX_TIME" --threads 10 --workers "$WORKERS" --refine-all --local-max-iter 5000 --local-tol 1e-8 --local-refine lj-newton --newton-linear-solver "$LINEAR" --newton-linear-ordering metis --ipopt-linear-solver mumps "${EXTRA[@]}" --output-dir "$OUT/LJ${N}/$LINEAR" 2>&1 | tee "$OUT/LJ${N}_${LINEAR}.log"
    cmp "$POP" "$OUT/LJ${N}/$LINEAR/LJ${N}_run1/initial.pfpop"
    jq '{algorithm,atoms,status,objective,seconds,objective_evaluations,local_calls,local_improvements,local_converged,local_limited,local_failed,linear_metis_analyses,mumps_user_ordering_analyses,linear_cg_fallbacks,ordering:.options.newton_linear_ordering}' "$OUT/LJ${N}/$LINEAR/LJ${N}_run1/pf.json"
  done
  jq '{algorithm,atoms,status,objective,seconds,objective_evaluations}' "$OUT/LJ${N}/${BACKENDS[0]}/LJ${N}_run1/ipopt_multistart.json"
done
printf 'RESULTATS=%s\n' "$OUT"
)
```

Changer `BACKENDS=(ldl)` ou `BACKENDS=(mumps)` permet un seul choix avec son
propre témoin Ipopt. Le raffinement supplémentaire de fin de boucle reste
limité à `refine-top-k` (5 par défaut), comme en 1.3.0.

## Autres commandes (modes historiques)

# Installer et tester PF C++ 1.3.1 — Linux, float64

L'archive est placée dans `/data/projects/PF_GlobalOptimization`.
Toutes les opérations numériques sont en `double` IEEE 754 binary64 ; il n'y a
pas de mode float32. `--version` affiche `float64=1`.
Les commandes suivantes utilisent l'environnement existant `pf_cpp_110`.

## Installation, compilation et tests

```bash
(
set -Eeo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pf_cpp_110
conda install --yes --channel conda-forge cmake ninja cxx-compiler pkg-config \
  "eigen=3.4" "ipopt=3.14" mumps-seq mumps-include \
  "python=3.12" "numpy=2.3.5" scipy scikit-learn jq
BASE=/data/projects/PF_GlobalOptimization
ARCHIVE="$BASE/pf_cpp_v1.3.1_source.tar.gz"
SRC="$BASE/pf_cpp_v1.3.1"
BUILD="$SRC/build_release"
test -s "$ARCHIVE"
test ! -e "$SRC"
tar -xzf "$ARCHIVE" -C "$BASE"
cd "$SRC"
sha256sum --quiet -c MANIFEST.sha256
unset LD_PRELOAD ASAN_OPTIONS UBSAN_OPTIONS LSAN_OPTIONS
export OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DPython3_EXECUTABLE="$CONDA_PREFIX/bin/python" \
  -DPF_NATIVE=ON -DPF_WITH_OPENMP=ON -DPF_WITH_IPOPT=ON -DPF_WITH_MUMPS=ON \
  -DPF_MUMPS_INCLUDE="$CONDA_PREFIX/include" \
  -DPF_MUMPS_LIBRARY="$CONDA_PREFIX/lib/libdmumps_seq.so" \
  -DBUILD_TESTING=ON
cmake --build "$BUILD" --parallel 8
ctest --test-dir "$BUILD" --output-on-failure
"$BUILD/pf_solver" --version
"$BUILD/pf_linear_check" --backend ldl
"$BUILD/pf_linear_check" --backend mumps
"$BUILD/pf_linear_check" --backend ipopt-mumps
)
```

La commande refuse d'écraser un dossier source préexistant. Pour reconstruire
une extraction déjà vérifiée, reprendre à `cmake` avec les mêmes variables.
`mumps-seq` fournit le MUMPS séquentiel, et `mumps-include` ses en-têtes.
Ne pas substituer une bibliothèque MPI sans adapter l'initialisation MPI.
Les chemins explicites empêchent de mélanger des en-têtes et une autre version
de MUMPS déjà présente sur la machine.

## Test PF sur 38 et 76 atomes, puis multistart Ipopt

```bash
(
set -Eeo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pf_cpp_110
BASE=/data/projects/PF_GlobalOptimization
BUILD="$BASE/pf_cpp_v1.3.1/build_release"
OUT="$BASE/results_pf130_$(date +%Y%m%d_%H%M%S)"
ATOMS=(38 76)
RUNS=1
THREADS=4
WORKERS=4
NEWTON_LINEAR=cg                 # cg, ldl, mumps ; ma27 nécessite HSL
IPOPT_LINEAR=mumps              # mumps ou ma27
export OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
"$BUILD/pf_solver" --atoms "${ATOMS[@]}" --runs "$RUNS" --seed 2024 \
  --max-iter 1000 --no-improve-stop 400 --max-time 3600 \
  --threads "$THREADS" --workers "$WORKERS" --refine-top-k 5 \
  --local-max-iter 5000 --local-tol 1e-8 \
  --local-refine lj-newton --newton-linear-solver "$NEWTON_LINEAR" \
  --ipopt-linear-solver "$IPOPT_LINEAR" --include-ipopt --output-dir "$OUT"
jq '{algorithm,atoms,status,objective,seconds,objective_evaluations,linear_factorizations}' \
  "$OUT"/LJ*/pf.json "$OUT"/LJ*/ipopt_multistart.json
)
```

Ne pas fixer `--particles` pour garder la règle Python `max(3N+56,50)` :
170 particules pour LJ38, 284 pour LJ76. Passer `RUNS=30` pour une campagne.
Le budget de 3600 secondes s'applique séparément à chaque méthode et chaque run.
`--include-ipopt` ajoute un multistart Ipopt depuis la même population fournie ;
ce n'est pas PF avec raffinements Ipopt. Le seuil de stagnation 400 est explicite,
`--no-improve-stop 0` le désactive. Aucune valeur ne garantit l'optimum LJ76.

## Comparer les solveurs locaux sur les mêmes points

```bash
(
set -Eeo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pf_cpp_110
BASE=/data/projects/PF_GlobalOptimization
BUILD="$BASE/pf_cpp_v1.3.1/build_release"
OUT="$BASE/bench_local130_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
for N in 38 76; do
  for LINEAR in cg ldl mumps; do
    INPUT=()
    if [ "$LINEAR" != cg ]; then
      INPUT=(--population "$OUT/LJ${N}_cg.json.initial.pfpop")
    fi
    "$BUILD/pf_local_bench" --atoms "$N" --starts 8 --workers 4 \
      --method lj-newton --newton-linear-solver "$LINEAR" \
      --local-tol 1e-8 --local-max-iter 5000 "${INPUT[@]}" \
      --output "$OUT/LJ${N}_${LINEAR}.json"
  done
  "$BUILD/pf_local_bench" --atoms "$N" --starts 8 --workers 4 \
    --method ipopt --ipopt-linear-solver mumps \
    --local-tol 1e-8 --local-max-iter 5000 \
    --population "$OUT/LJ${N}_cg.json.initial.pfpop" --output "$OUT/LJ${N}_ipopt.json"
done
jq '{atoms,method,newton_linear_solver,workers,core_seconds,audit_pass_count,starts,linear_cg_fallbacks}' "$OUT"/*.json
)
```

Les fichiers `.initial.pfpop` et `.final.pfpop` conservent tous les points.
La projection native du départ hors boule et la procédure interne d'Ipopt
peuvent différer ; les coordonnées fournies sont rigoureusement identiques.
L'audit compare faisabilité et `||x-P(x-gradient)||_inf <= 1e-8` après optimisation.
Le temps seul ne suffit pas : comparer aussi les énergies et cet audit.
Pour répéter le protocole à trois répétitions avec CG et Ipopt, séquentiel et parallèle :

```bash
python /data/projects/PF_GlobalOptimization/pf_cpp_v1.3.1/tests/benchmark_local.py \
  /data/projects/PF_GlobalOptimization/pf_cpp_v1.3.1/build_release/pf_local_bench \
  --atoms 38 76 --starts 8 --repeats 3 --workers 4 \
  --output "/data/projects/PF_GlobalOptimization/bench_repeated_$(date +%Y%m%d_%H%M%S)"
```

## MA27 avec la bibliothèque HSL de l'utilisateur

HSL n'est pas distribué dans cette archive. Fournir une bibliothèque partagée
MA27 **double précision, INTEGER Fortran 32 bits**, exportant `ma27id_`,
`ma27ad_`, `ma27bd_`, `ma27cd_`. Adapter `HSL` à son chemin réel.
La présence des symboles est contrôlée ; l'ABI de compilation doit correspondre.

```bash
(
set -Eeo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pf_cpp_110
BASE=/data/projects/PF_GlobalOptimization
BUILD="$BASE/pf_cpp_v1.3.1/build_release"
HSL="$CONDA_PREFIX/lib/libcoinhsl.so"
test -r "$HSL"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
"$BUILD/pf_linear_check" --backend ma27 --hsl-library "$HSL"
"$BUILD/pf_linear_check" --backend ipopt-ma27 --hsl-library "$HSL"
"$BUILD/pf_solver" --atoms 38 76 --runs 1 --seed 2024 \
  --max-iter 1000 --no-improve-stop 400 --max-time 3600 \
  --threads 4 --workers 4 --refine-top-k 5 --local-max-iter 5000 --local-tol 1e-8 \
  --local-refine lj-newton --newton-linear-solver ma27 \
  --include-ipopt --ipopt-linear-solver ma27 --hsl-library "$HSL" \
  --output-dir "$BASE/PF_MA27_$(date +%Y%m%d_%H%M%S)"
)
```

Ipopt doit avoir été compilé avec MA27 ou son chargeur dynamique HSL. Un échec
est explicite ; aucune substitution automatique à MUMPS n'est effectuée.
`--ipopt-linear-solver ldl` n'est pas disponible : `ldl` désigne ici Eigen LDLT
pour le solveur natif. Les [options officielles d'Ipopt](https://coin-or.github.io/Ipopt/OPTIONS.html)
exposent notamment MA27/MUMPS, mais pas un backend autonome nommé `ldl`.

## Compilation indépendante d'Ipopt et de MUMPS

```bash
(
set -Eeo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pf_cpp_110
SRC=/data/projects/PF_GlobalOptimization/pf_cpp_v1.3.1
cmake -S "$SRC" -B "$SRC/build_portable" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DPython3_EXECUTABLE="$CONDA_PREFIX/bin/python" \
  -DPF_NATIVE=OFF -DPF_WITH_OPENMP=ON -DPF_WITH_IPOPT=OFF -DPF_WITH_MUMPS=OFF
cmake --build "$SRC/build_portable" --parallel 8
ctest --test-dir "$SRC/build_portable" --output-on-failure
"$SRC/build_portable/pf_solver" --version
)
```

CG et LDL sont disponibles dans ce build. Sélectionner explicitement
`--local-refine lj-newton` ; ne pas demander `--include-ipopt`.
