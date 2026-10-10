# Raffinement LJ spécialisé — Newton à région de confiance

Le backend `--local-refine lj-newton` remplace les appels locaux à Ipopt par
une implémentation native spécialisée pour Lennard–Jones en dimension 3.
Il n'appelle ni Ipopt ni un solveur quasi-Newton. Ipopt reste disponible pour
les comparaisons, et demeure le défaut historique si aucune méthode n'est passée.

## Dérivées exactes et structure par paires

Avec d_ij=x_i-x_j et s_ij=||d_ij||²+10^-8, l'énergie est

    E(x) = somme_{i>j} [s_ij^-6 - 2 s_ij^-3].

Pour une paire, définir

    a_ij = 12 (s_ij^-4 - s_ij^-7),
    b_ij = 168 s_ij^-8 - 96 s_ij^-5.

Sa contribution au gradient est a_ij d_ij pour i et son opposé pour j.
Sa courbure est le bloc exact K_ij=a_ij I+b_ij d_ij d_ij^T.
Pour tout vecteur v, le produit exact est accumulé comme suit :

    (Hv)_i += a_ij(v_i-v_j) + b_ij d_ij [d_ij^T(v_i-v_j)],
    (Hv)_j -= la même quantité.

Les déplacements d_ij et les deux coefficients sont mis en cache une seule
fois par itéré accepté, en même temps que le gradient. Les rejets réutilisent
ce cache. Aucun calcul par différences finies, BFGS ou approximation du potentiel
n'est utilisé. Le produit coûte O(N²), avec stockage O(N²), sans factorisation
d'une matrice dense 3N par 3N ni assemblage d'un système KKT avec variables de barrière.

## Pas de Newton et globalisation

Avec `--newton-linear-solver cg` (défaut), le modèle quadratique est traité par gradient conjugué tronqué, préconditionné,
dans une région de confiance. Une courbure négative provoque une sortie sur la
frontière de la région. Le préconditionneur est diagonal par atome, à partir de
la somme des bornes positives |a_ij|+|b_ij| ||d_ij||² des blocs exacts.
Il ne remplace pas la Hessienne dans le modèle.

Un pas de Cauchy projeté fournit une alternative lorsque le pas de Newton
prédit moins de réduction. L'acceptation utilise la réduction effective de
l'énergie rapportée à la réduction prédite. La région rétrécit en cas de
mauvais accord et s'agrandit lorsque le modèle est fiable sur sa frontière.
Au niveau de l'arrondi machine, une amélioration du résidu projeté peut accepter
un pas dont la variation d'énergie n'est plus résolue en double précision.

Les translations rigides donnent des directions nulles de la Hessienne :
l'algorithme n'a pas besoin d'inverser cette matrice singulière. Les coïncidences
exactes entre atomes, qui peuvent être des maxima stationnaires du potentiel
lissé, sont traitées par une direction antisymétrique vérifiée de courbure négative.
L'arrêt ordinaire teste une stationnarité du premier ordre ; il ne certifie pas
la positivité de toute la Hessienne ni l'optimalité globale.

La méthode s'appuie sur les principes classiques de Newton-CG avec région de
confiance, notamment Steihaug, *The Conjugate Gradient Method and Trust Regions
in Large Scale Optimization*, SIAM J. Numer. Anal. 20(3), 626–637 (1983),
https://doi.org/10.1137/0720042. La spécialisation aux noyaux et contraintes LJ
est implémentée ici ; cette livraison ne revendique pas une nouvelle méthode
de région de confiance ni un théorème de convergence propre à cette implémentation.

## Factorisations directes et précision

`--newton-linear-solver ldl|ma27|mumps` assemble la Hessienne exacte à partir
des mêmes blocs. Le modèle tangent est `P(H+2Λ)P`, complété par une courbure
positive sur les normales exclues. Une régularisation diagonale adaptative
rend le système défini positif, y compris en présence des modes rigides.
La direction résolue est tronquée à la région de confiance puis projetée.
Ce chemin est un Newton régularisé globalisé ; ce n'est pas une résolution
exacte du sous-problème de région de confiance.

- `ldl` : Eigen LDLT dense, inclus dans le projet.
- `ma27` : interface dynamique HSL double précision / entiers 32 bits.
- `mumps` : interface DMUMPS double précision / entiers 32 bits, compilation
  optionnelle `PF_WITH_MUMPS=ON`, bibliothèque séquentielle.

Les backends MA27 et MUMPS réutilisent l'analyse symbolique pendant un
raffinement ; les valeurs sont refactorisées pour chaque modèle/décalage.
Le signe des pivots (LDL) ou l'inertie (MA27/MUMPS) contrôle la positivité.
Chaque solution est vérifiée par son résidu. Après huit tentatives infructueuses,
un pas CG de secours est compté dans `linear_cg_fallbacks` ; une bibliothèque
absente ou une erreur de configuration provoque une erreur, jamais ce secours.
Les compteurs `linear_analyses`, `linear_factorizations`, `linear_solves`
permettent de vérifier quel travail a été réellement effectué.
La Hessienne LJ est structurellement dense : MA27/MUMPS ne sont pas supposés
plus rapides que CG ici. Leur intérêt doit être mesuré sur les cas cibles.

Toute l'arithmétique du portage utilise `double`, équivalent IEEE 754 binary64
au `float64` de NumPy : coordonnées, poids, énergie, gradients, Hessiennes,
produits H*v, facteurs, contraintes, tolérances et échanges entre workers.
Des assertions de compilation imposent 64 bits, mantisse 53 bits, IEEE 754,
et les scalaires `double` d'Eigen, d'Ipopt et de DMUMPS. `fast-math` est
interdit et la contraction FMA désactivée. Les sorties conservent 17 chiffres
significatifs ; graines, indices et compteurs restent des entiers.
Les calculs du Python archivé en float32 ne sont pas modifiés : il est conservé
comme preuve de provenance et n'est pas appelé par le solveur C++.

## Contraintes identiques à celles de la comparaison Ipopt

Par défaut, chaque atome satisfait ||x_i||² <= R², centré à l'origine, et les
bornes sont respectées. Le rayon actif est transmis après les kicks PF.
Le backend exige que ces boules soient contenues dans la boîte, ce qui est
le cas des domaines CLI et des rayons contractés utilisés dans PF.
Une géométrie différente est rejetée explicitement.

La projection sur chaque boule est exacte. Pour une boule active dont la force
pointe vers l'extérieur, le CG travaille dans le plan tangent. Si lambda_i
désigne le multiplicateur estimé de ||x_i||²-R², alors

    lambda_i = max(0, -x_i^T g_i / (2 ||x_i||²)),
    Hessienne du Lagrangien = H + diag_i(2 lambda_i I).

La courbure des contraintes est ainsi incluse dans le sous-problème tangent.
L'essai est reprojeté sur le domaine avant son évaluation. La diminution prédite
est calculée sur le déplacement projeté avec la Hessienne de l'objectif.
Le pas de Cauchy projeté reste disponible lorsque l'ensemble actif change.

Un départ situé hors des boules est projeté avant les itérations natives.
Ipopt traite le même point fourni avec sa propre procédure d'initialisation ;
les points fournis sont identiques, les mécanismes internes de faisabilité diffèrent.
`--no-local-ball` (alias de `--no-ipopt-ball`) sélectionne explicitement les
bornes seules ; aucune suppression implicite des contraintes n'a lieu.

## Parallélisme

Avec `lj-newton` et `cg` ou `ldl`, `--workers W` distribue les raffinements indépendants sur W
threads OpenMP. Chaque tâche possède son contexte, ses buffers, son cache de
courbure et ses compteurs. Les générateurs PF restent sur le coordinateur.
Les résultats sont fusionnés dans l'ordre des particules, indépendamment de
l'ordonnancement. Il n'y a ni processus enfant, ni socket, ni appel à MUMPS.

Avec les backends natifs MA27/MUMPS et avec Ipopt, le même paramètre utilise les processus persistants isolés,
nécessaires pour ne pas partager l'état d'Ipopt/MUMPS entre threads.
`--threads T` concerne les phases globales PF. Les phases locales et globales
sont successives, afin d'éviter leur imbrication. Avec top-k=5, davantage de
cinq workers n'accélère pas ce lot.

Le budget d'évaluations est réservé par tâche, puis agrégé sans dépassement.
Les délais sont vérifiés dans la boucle Newton et dans le CG. Comme pour Ipopt,
un noyau élémentaire déjà engagé n'est pas préempté. Une limite contraignante
peut changer les raffinements terminés quand le nombre de workers change.

## Interface et mesure

```bash
./build_release/pf_solver --atoms 38 76 --local-refine lj-newton --workers 4 --threads 4 --include-ipopt --output-dir results_newton
```

Paramètres : `--local-tol`, `--local-max-iter`, `--newton-cg-max-iter` (0 : 3N),
`--newton-radius` (rayon initial divisé par sqrt(N), défaut 0.2).
Les JSON ajoutent `hessian_products`, `newton_iterations`, `newton_accepted`,
`newton_rejected`, `negative_curvature` et le mode d'exécution locale.
Une évaluation de Hessienne native correspond à une mise à jour de ses
coefficients exacts ; les applications successives H*v sont comptées séparément.

Le programme `pf_local_bench` compare les seuls raffinements. Il conserve
la population initiale et chaque résultat, sépare préparation et calcul,
et audite indépendamment la faisabilité et le résidu non mis à l'échelle

    ||x - Projection_C(x - gradient E(x))||_inf.

Ce résidu tient compte des boules, contrairement au diagnostic historique
`projected_gradient_inf` du résultat PF, qui ne concerne que la boîte.
Le test d'arrêt interne d'Ipopt et celui du Newton natif diffèrent : le rapport
publie donc aussi le nombre de points passant ce même audit indépendant.
Les minima locaux peuvent différer ; comparer temps, qualité et audits ensemble.
Les audits sont hors chronométrage et hors budget d'optimisation, avec des
compteurs séparés. Les entrées sorties ainsi que la génération des départs
sont également hors du temps `core_seconds`.

Le backend natif fonctionne avec `PF_WITH_IPOPT=OFF`. Pour comparer à Ipopt,
compiler avec `PF_WITH_IPOPT=ON`. Les commandes complètes pour la machine cible
et les résultats mesurés figurent dans `INSTALL_TEST_COMMANDS.md` et `VALIDATION.md`.
