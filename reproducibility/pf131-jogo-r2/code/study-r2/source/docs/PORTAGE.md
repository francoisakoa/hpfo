# Portage de ReleasePFJOGO — base conservée dans PF C++ 1.3.0

Cette note décrit le parcours Python/Ipopt porté jusqu’à 1.2.0. La 1.3.0
ajoute le solveur local spécialisé décrit dans `LJ_NEWTON.md` ; les garanties
de précision et les tests actuels sont dans `VALIDATION.md`.

La v1.2.0 aligne les graines, PCG64 et les départs sur NumPy.
Elle conserve le correctif mémoire Eigen de la v1.1.1.
Le protocole exact et ses limites figurent dans `RANDOM_PROTOCOL.md`.

La référence est `ReleasePFJOGO-v1.0-jogo-r2.tar.gz`. Les quatre modules Python
actifs nécessaires à cette comparaison sont conservés octet pour octet dans
`reference/`. Leurs empreintes figurent dans `reference/SOURCES.json`.
La source précédente reste dans `reference/legacy_v1/` pour tracer les changements.

## Ce que les trois modules reçus ont permis de corriger

| Fichier reçu | Comportement vérifié | Correction C++ 1.1.0 |
|---|---|---|
| `solve_nlp_local_gpu.py` | JAX et NumPy utilisent s=distance²+1e-8. | Même énergie lisse, gradient et Hessienne ; suppression du clamp de distance adopté en v1.0 faute du module. |
| `solve_nlp_ipopt.py` | La fonction active `solver_lj_cyipopt` construit un modèle Pyomo avec bornes et une boule par atome. | Contraintes natives Ipopt, Jacobienne creuse à 3N coefficients en 3D, Hessienne du Lagrangien avec 2*lambda sur les diagonales atomiques. |
| `backend_ops.py` | Sharing incluant soi-même, diversité moyenne, K-means CPU `n_init="auto"`, GMM complet, DBSCAN. | Sharing exact ; K-means++ glouton avec un redémarrage par défaut, 300 itérations et tolérance 1e-4 pondérée par la variance ; GMM à 100 itérations. |
| `general_nlp_pf_optimized_gpu.py` actualisé | Stagnation globale persistante, `searchsorted(side="left")`, nouveaux réglages par défaut et références LJ précises. | Arrêt `stagnation_limit`, frontières du rééchantillonnage et sélection conditionnelle de N0 alignées ; max_iterations=1000, max_seconds=20000. |

L'énergie et ses dérivées sont comparées directement aux fonctions du module
JAX fourni et aux expressions du modèle Pyomo fourni. Le test de construction
Pyomo capture le modèle sans appeler son exécutable Ipopt : il n'est pas un
rejeu d'optimisation Python. Des essais C++ distincts vérifient les solutions
locales contre l'optimum analytique de deux atomes dans des boules actives.

## Contraintes, faisabilité et incumbent

Les contraintes sont **une boule centrée à l'origine pour chaque atome**, et
non une unique boule dans R^(3N), ni une boule centrée sur le point initial.
Pour chaque atome a : g_a(x)=sum_k(x[a*d+k]^2) <= R². La Jacobienne a pour
coefficient 2*x[a*d+k]. La Hessienne est
`objective_factor * Hessian(E) + diag(2*lambda[atom])`.
La borne inférieure de g est absente, comme dans Pyomo.

Le rayon effectif est transmis à chaque tâche des processus persistants, y
compris après un kick et pendant le raffinement final. Les essais locaux hors
boules ne deviennent pas de meilleurs points locaux. Un point initial hors
boules peut avoir une énergie plus basse qu'une solution faisable : il ne doit
pas annuler cette solution au seul motif que l'énergie augmente.
La tolérance de sélection faisable sur g est `1e-10*max(1,R²)`.

Dans la source, les phases globales PF n'imposent que la boîte. Un incumbent PF
issu d'une particule non raffinée peut donc être hors des boules. Le portage
conserve cette distinction et publie le rayon atomique maximal du résultat.
Le multistart Ipopt, lui, retient exclusivement des points satisfaisant les
boules. La boîte originale est toujours respectée. `--no-ipopt-ball` désactive
explicitement les boules pour une expérience d'ablation.

La routine Python locale renvoie un objectif factice 0.0 ; la boucle PF le
réévalue. Elle peut également retourner le point initial après une exception,
avec le même 0.0 : le repli L-BFGS-B annoncé dans un message est commenté dans
la fonction active. Le portage utilise des couples coordonnées/énergie vérifiés,
compte les échecs et refuse un backend Ipopt absent ou mal configuré.

## Correspondance de la boucle PF

| Source Python | C++ | Conservation / précision |
|---|---|---|
| `make_lj_objective` et objectif Pyomo actif | `LennardJones` | Somme de s^-6 − 2s^-3, s=distance²+1e-8. Sommation compensée ; gradient/Hessienne analytiques. |
| `_random_box_point` | `uniform_population` | Uniforme sur la boîte finie ; les bornes infinies sont explicitement rejetées. |
| Réduction du nombre de particules | `Solver::solve` | max(min_particles, int(N0*(1-t/T))) ; suppression aléatoire sans remise. |
| `invertible_particle_flow` | `flow_population` | Groupes, choix du meilleur, alpha=0.98*(1-t/T)+0.05, projection puis jitter 0.02*(ub-lb). |
| `fast_particle_flow` | `flow_population` | Même attraction, jitter 0.04*(ub-lb). |
| `linear_mapping` | `map_particle` | Formule identique ; alpha vaut **1.03** au premier tour, conformément au code. |
| `residual_mapping` | `map_particle` | x+alpha*tanh(best-x). |
| `affine_coupling_mapping` | `map_particle` | Formule conservée en dimension paire ; dimension impaire rejetée car le Python a des tailles incompatibles. |
| `cnf_mapping` | `map_particle` | Même ODE, résolue analytiquement : best+asinh(exp(-alpha)*sinh(x-best)). Comparée à une intégration précise. |
| `_mutate_particles_adaptive`, `mutate_wrapper` | `flow_population` | Branches de probabilités 0.4/0.7/0.85/0.95, différence F=.8, mélange et archive ; covariance empirique optionnelle. |
| `_compute_weights` | `compute_weights` | Objectif effectif, écart-type, exponentielle, plancher 1e-8, normalisation, ESS. Stabilisation par changement d'échelle. |
| `compute_novelty` | `novelty_scores` | Distance moyenne aux éléments d'une archive des 10 améliorations récentes. |
| `_resample_particles_by_weights` | `stratified_indices` + jitter | Rééchantillonnage **stratifié**, CDF avec côté gauche aux frontières, un uniforme par strate ; jitter .01*(ub-lb). |
| ESS et injections | `Solver::solve` | Seuils .1*N0 et .05*N0, injection .1*N0, puis rétablissement de la taille N0. |
| `adjust_beta` | `next_beta` | Même formule et plafond **5**, pas 5*log(1+N). Cas T=1 sécurisé. |
| `island_exchange` | `Solver::solve` | Tous les 10 tours : 3 îlots, ajout des meilleurs et sélection de N0 particules. |
| Représentants par K-means | `Solver::solve` | Tous les 20 tours, 3 groupes et ajout des représentants. |
| `tune_parameters` | `Solver::solve` | Adaptation sigma, facteur beta et injection de .2*N0 si diversité faible. |
| Kicks | `Solver::solve` | Après 10 stagnations, amplitude, facteur .38 et contraction du rayon conformes. Le domaine original reste une limite extérieure. |
| `_final_local_optimization` | `Solver::solve` | Raffinement final top-k conservé, sous budget ; supprimé si cible déjà atteinte. |


## Écarts intentionnels et portée de la fidélité

- Calcul natif double, sans JAX/CUDA/cuML à l'exécution. Le parcours principal
  de cette release est PF + Ipopt. Les options JAX PGD, Adam et L-BFGS-B ne sont
  pas portées ; `projected-lbfgs` est une alternative différente, sur la boîte.
- Les mêmes équations de clustering sont implémentées nativement. Les tirages,
  départages d'égalités et réparations de groupes vides peuvent différer de
  scikit-learn/cuML ; les partitions ne sont pas certifiées identiques. Le choix
  K-means++ glouton et les réglages CPU sont documentés dans la
  [documentation officielle KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html).
- PCG64/SeedSequence et les lois utilisées sont alignés sur NumPy 2.3.5.
  Les graines et les coordonnées de départ sont vérifiées pour 30 runs de
  LJ38 et LJ76. Le filtre consomme un entier pour la graine du clustering,
  qui utilise ensuite un flux MT19937 de type RandomState indépendant.
- Les `default_rng()` sans graine des îlots et de l'injection ne permettent
  pas de reconstituer leurs tirages historiques. Le C++ leur attribue deux
  sous-séquences déterministes explicites, sans consommer le flux PF principal.
  Cela reste un écart avec la source ; aucune trajectoire LJ complète n'est certifiée.
- Tous les objectifs calculés sont comptés, y compris dans les workers. Les
  gradients/Hessiennes et les audits de sortie ont des compteurs séparés.
  La répartition d'un budget local contraignant peut modifier une trajectoire
  parallèle ; les seuils temporels aussi.
- Le meilleur point évalué est protégé dès l'initialisation et lors des essais
  locaux faisables. Cela peut améliorer l'incumbent ou déclencher la cible plus
  tôt que le Python. Aucun raffinement initial supplémentaire n'est introduit.
- Le calcul EnKF sans effet sur la population est retiré. Une proposition
  non évaluée ne peut pas devenir le résultat final.
- `adjust_beta` conserve le facteur de diversité effectif 1. La mutation
  conserve le terme dirigé nul associé à `self.best_particle` jamais renseigné.
- Les poids non finis sont traités explicitement. Le dénominateur des poids
  est la somme exacte au lieu de somme+1e-12. La CNF utilise la solution
  analytique de la même ODE, validée contre une intégration numérique précise.
- Les journaux conservent l'itération terminale, y compris lors d'un arrêt sur
  cible ou stagnation. Le raffinement final respecte les budgets et est omis
  lorsque la cible a déjà été atteinte.
- Le CLI lance par défaut LJ38 une fois, pour éviter une campagne implicite
  de 8 tailles x 30 runs. La commande complète est dans le README. Le nombre
  de particules CLI reste max(3*N+56,50).
- Les mécanismes neuronaux, les routines non appelées et les contraintes NLP
  arbitraires ne sont pas annoncés comme fonctionnalités prises en charge.

Les trois fichiers ne sont donc plus manquants. Cela permet une comparaison
mathématique directe, sans établir une identité bit à bit de toute la trajectoire
Python/GPU, ni certifier les résultats des tableaux de l'article.
L'atteinte d'une référence énergétique reste un critère empirique, sans certificat
d'optimalité globale.
