# Baselines v3.0.0 — corrections depuis la v2 fournie

Les familles BH-OJ, GA, PSO, CLPSO, SMC, les huit instances, les 30 runs et les
plafonds numériques restent conservés. Le code commun est refactorisé pour éviter
que CPU, GPU, runner et agrégateur divergent. Ce n'est pas un simple changement
cosmétique : les changements de protocole sont listés ci-dessous et empêchent le
mélange automatique des anciens et nouveaux résultats.

| Constat audit v2 | Correction v3 | Vérification livrée |
|---|---|---|
| Gradient = force opposée | Deux gradients analytiques corrects, selon potentiel explicite | Différences finies 2/5/13/38/98 atomes ; dimère relaxé à -1 |
| Potentiel v2 différent du C++ PF131 | Défaut `pf131` q3*(q3-2), q=1/(r2+1e-8) ; mode standard séparé | Oracle scalaire/Kahan, tests des deux modèles ; source C++ archivée |
| L-BFGS-B sans bounds puis clipping/res.fun périmé | Bornes transmises à SciPy ; retour du meilleur couple évalué | Vrai dimère et solveur substitué retournant volontairement res.fun incohérent |
| Meilleur point perdu à l'arrêt | Incumbent central mis à jour à chaque évaluation, partagé entre tentatives | Budgets 1/11/401, cinq méthodes, appels locaux et lots partiels |
| SMC sans mouvements à ESS .7 | Étape adaptative, rééchantillonnage et mouvements à chaque niveau | Compteurs de propositions, populations20, fin naturelle et interruptions |
| Adaptation et poids SMC incompatibles | Même pondération log-domain sans troncature des écarts | ESS vérifiée jusqu'à énergies d'ordre1e100 |
| Incrément beta forcé1e-6 | Recherche sans dimension et nextafter journalisé si nécessaire | ESS réelle conforme à la cible ou plafond atteint |
| CPU/GPU float32 et chemins distincts | Float64 ; logique et RNG communs ; seuls lots énergie sur CuPy | CPU validé ; test CuPy conditionnel ; précontrôle GPU obligatoire |
| Loi initiale sphérique annoncée box | Box uniforme sans transformation par défaut ; options alternatives explicites | Reproduction exacte du tirage PCG64 de l'initialisation |
| Seed premier essai caché | Premier essai = seed racine direct ; seeds de chaque restart consignés | LJ38/run1=2162819792 ; répétabilité et séquences de restarts |
| Paramètres ignorés | Toute clé inconnue refusée ; paramètres résolus copiés dans résultat | beta_min appliqué ; w CLPSO refusé ; mauvaises valeurs rejetées |
| CLPSO exemplaires figés comme coordonnées | Indices d'exemplaires suivis jusqu'au refresh | Répétabilité et essais aux huit tailles |
| CLPSO calendrier presque constant | Défaut selon budget consommé ; mode itérations explicite | Paramètres/règle dans identité ; changement documenté |
| Mutation GA hors convention bornée | Mutation polynomiale bornée explicite | Domaine et tests des cinq méthodes |
| Restart900s masquant SMC | Pas de timeout par tentative par défaut ; fin naturelle restartable | Test de reset deadline et conservation du budget |
| Plafond36000s interprété comme budget épuisé | time_limit distinct ; censures séparées ; coopératif clairement indiqué | Agrégation bloque par défaut, opt-in explicite avec effectifs |
| Erreur numérique redémarrée silencieusement | Sauvegarde d'erreur et retour non nul | Injection contrôlée d'exception |
| Champs runner/agrégateur incompatibles | Schéma unique contrôlé et checksum | Succès, stop reasons, paramètres, compteurs, coordonnées et énergie vérifiés |
| Doublons/incomplets acceptés | Identités attendues uniques et validation exhaustive | Doublons, JSON malformés, faux succès et budgets mélangés rejetés |
| Résumé périmé exportable | Hash des fichiers bruts vérifié avant LaTeX | Modification brute post-agrégation détectée |
| SD population/médiane supérieure | SD n-1 ; médiane usuelle ; échecs inclus | Méthodes NumPy explicites et contenu des résumés |
| Reprise ignore changement budget/code | Empreinte code/config/env ; archivage sous --force | 2000→autre budget refusé (test utilise de petits budgets) ; résultat identique sauté |
| Lanceur retour0 malgré enfant en échec | Code non nul si enfant échoue ; logs conservés | Enfant test retournant17 |
| nproc-1 jobs implicitement sur GPU | CPU4jobs par défaut ; GPU1, partage explicite | Contrôle des arguments et exécution parallèle40jobs techniques |
| Ancien driver divergeant du runner | `benchmark_suite.py` délègue à la même implémentation | Aucun second chemin de comptage indépendant |

## Ce qui n'est PAS revendiqué

- Pas de campagne 1200 runs aux budgets de production effectuée dans cet environnement.
- Pas d'exécution CUDA validée ici si CuPy est absent (voir le rapport).
- Pas de preuve de supériorité de HPF ni de taux de succès attendus des baselines.
- Pas d'identité des populations initiales avec les fichiers PF131 non fournis.
- Pas de reprise bit à bit au milieu d'un run interrompu.
- Pas de modification du manuscrit, des figures, de ses données ou du solveur C++.

La formule C++ est une information nouvelle issue du source reçu après l'audit v2.
Elle est corrigée avant la campagne plutôt que de comparer silencieusement deux
oracles différents aux mêmes coordonnées. La configuration et tous les résultats
portent le nom du potentiel effectivement utilisé.
