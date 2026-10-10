# Protocole d'ablation et de sensibilité — LJ75 et LJ98

## Portée et identité scientifique

Le plan reproduit les **lignes et les grilles** des tableaux 7, 8, 9 et 10 de
`PFO_LJ_JOGO_SN_R2(1).pdf`, pages 20–21. Les nouvelles mesures sont celles d'une
**implémentation C++ PF131 instrumentée pour cette étude**, pas des fichiers bruts
retrouvés de l'expérience Python historique, ni une promesse d'en retrouver les
pourcentages. Aucune ancienne valeur des tableaux n'est injectée dans les résultats.
LJ77 est exclu. L'installation et les 240 résultats PF131 archivés sont inchangés.

La source originale distribuée est conservée dans `reference/pf_cpp_v1.3.1_source.tar.gz`.
La copie de travail est dans `source/`, les changements dans `reference/PF131_ARTICLE_STUDY.patch`.
Le binaire reçoit la version distincte `1.3.1-article-study-r1`.

## Table 7 — 30 runs par ligne et par taille

|Configuration|Intervention réelle dans la copie PF131|
|---|---|
|Full HPF|Contrôle complet de cette étude : paramètres ci-dessous|
|− IPF operator|Transport désactivé, sans basculer vers une autre mutation ; ordre par clusters et jitter conservés|
|− Sharing metric|Poids sharing nul et calcul de la métrique omis|
|− Novelty metric|Poids novelty nul et calcul de la métrique omis|
|− Kick mechanism|Perturbation de kick et adaptation du rayon associée désactivées ; reste de la gestion de stagnation conservé|
|− Island exchange|Seul échange périodique toutes les 10 itérations désactivé ; injection périodique toutes les 20 conservée|
|− Local optimization|Aucun appel local, ni pendant la boucle ni au raffinement final|
|− Adaptive annealing|Mise à jour de beta et de son facteur désactivée ; beta constant = 1e-4 par défaut|
|Only PF + local opt (no diversity)|Sans IPF, sharing, novelty, kicks, échanges/injections d'îles, injection de diversité et réinitialisation ESS ; raffinement local conservé|
|Only PF (no local, no diversity)|Configuration précédente sans aucun raffinement local|

Les deux dernières lignes conservent le bruit d'exploration ordinaire, le
rééchantillonnage et le recuit, qui constituent ici le noyau PF réduit. Les calculs
purement diagnostiques de diversité peuvent rester présents : ils ne modifient pas
la population.

**Conventions à déclarer dans la ressoumission.** L'article ne précise pas la valeur
fixe de beta dans l'ablation du recuit ni la liste exacte des mécanismes de ses deux
lignes réduites. Les conventions opérationnelles ci-dessus sont donc nouvelles et
explicites, pas des détails historiques supposés retrouvés. `--fixed-beta` permet de
changer cette convention, dans un nouveau dossier de campagne, sans retoucher le C++.
Une telle décision doit être prise avant examen des résultats et conservée dans le protocole.

## Tables 8–10 — 10 runs par valeur et par taille

* Table 8 : nombre **initial** de particules 100, 200, 300, 400, 500.
* Table 9 : beta initial 1e-5, 1e-4, 1e-3, 1e-2.
* Table 10 : véritable alpha initial 0.95, 0.98, 0.99, 1.00 ; alpha final 0.05.

Une seule valeur est changée à la fois, tous les autres réglages sont ceux du contrôle.
Les lignes beta=1e-4 et alpha=0.98 utilisent exactement les runs 1–10 du contrôle
Full HPF. Ce partage est défini **avant** l'exécution, sans sélection des meilleurs runs.
Il donne 860 observations dans les tables, mais **820 calculs distincts** :
600 pour la table 7 et 220 calculs supplémentaires pour les tables 8–10.
Chaque ligne de `reports/RUNS.csv` contient l'identifiant du calcul et le chemin brut
réel ; aucune copie n'est présentée comme un run indépendant.

## Contrôle commun de l'étude

* Tailles : LJ75 et LJ98, et aucune autre dans le lanceur de production.
* Np par défaut : max(3N+56,50), soit 281 et 350.
* Raffinement : **top 5**, tolérance 1e-8, maximum local 5000, Ipopt/MUMPS.
* Itérations PF : 1000 ; arrêt stagnation désactivé ; plafond mural : 36000 secondes.
* Pas de plafond d'évaluations par défaut (`max_evals=0`), comme protocole à plafond
  d'itérations. Les évaluations sont mesurées, mais ne sont pas artificiellement égalisées.
* Succès énergétique : objectif <= cible PF131 + 1e-4, indépendamment de la convergence
  locale. Les coordonnées et la norme de gradient enregistrées restent consultables.
* Graines : base 2024 ; MD5 de `base_N_run_offset`, premiers 8 chiffres hexadécimaux,
  offsets 0 et 1000, protocole original PF131.
* Parallélisme : un seul run actif ; 10 threads batch et 8 workers locaux au maximum,
  bibliothèques BLAS à un thread. Pas de partage de cœurs entre deux runs de ce lanceur.
* Population : generation native PF131 puis gel par (taille, Np, numéro du run),
  réutilisée octet pour octet entre variantes. **Elle change entre runs**. Ce n'est pas
  le protocole de la campagne historique à une population figée par taille.
* Dans PF131, la population provient du flux `seed_pf`, pas du `seed_init` de x0 ;
  x0 est une métadonnée inutilisée par la boucle. Le script ne prétend pas le contraire.
* En faisant varier Np, les populations natives du même run ont des préfixes communs,
  mais la position du flux aléatoire au début de PF dépend de Np : pas de promesse de
  tirages ultérieurs identiques entre tailles de population.

## Alpha : intervention nécessaire, jamais cachée

PF131 original utilise `0.98*(1-t/T)+0.05`, donc 1.03 au premier transport.
Cette formule ne peut pas servir à étiqueter correctement la sensibilité alpha0
0.95/0.98/0.99/1.00. Dans **toutes** les configurations de cette étude, le transport
utilise `alpha0 + (0.05-alpha0)*t/(T-1)` : départ et arrivée exacts.
Le mode original reste disponible par défaut dans le binaire pour les tests de
non-régression, mais le lanceur demande explicitement le mode d'étude.

Alpha0=1 est volontairement non inversible : le transport rassemble les particules
sur l'élite **de leur cluster**, puis le jitter conservé les sépare. On ne prétend pas
que toute la population devient et reste globalement identique.

## Différences connues entre le manuscrit et le noyau PF131

Le présent kit n'est **pas** une réécriture silencieuse de tout le solveur pour le
rendre identique aux équations du manuscrit. Il expose les interventions de l'étude
sur le code réellement archivé. À expliciter dans le manuscrit révisé :

|Point|PF131 utilisé ici|
|---|---|
|Transport IPF|Élite par cluster, suivi de jitter ; pas uniquement une élite globale|
|Recuit adaptatif|Règle `next_beta` de PF131, différente de la récurrence (22)–(23) du texte|
|Sharing|Coefficient PF131 fixe 0.01, pas la formule dynamique du tableau 1|
|Archive de novelty|Maximum 10 entrées dans PF131, contre 20 dans le tableau 1|
|Population|Nombre courant décroissant, avec plancher et injections ; Np désigne le nombre initial|
|Kicks / rayon|Formule PF131 et rayon réduit vers 3, différents de la formule et de l'expansion exposées dans le texte|
|Échanges / injection|Opérations PF131, pas une réimplémentation des descriptions par centroïdes du texte|
|Raffinement|Ipopt/MUMPS, contraintes de boules natives ; pas MA27 avec pénalité ni fallback L-BFGS-B historique|
|Énergie|Potentiel PF131 `q^3(q^3-2)`, q=1/(r²+1e-8)|
|Matériel|C++ CPU, aucune accélération GPU|

Le réglage top-5 du tableau 1 et les alpha réellement initiaux sont employés, mais
ces autres différences ne sont pas dissimulées. Les 240 résultats précédemment
archivés (refine-all, populations communes et alpha original) **ne sont pas réutilisés
comme contrôle**. Le contrôle de cette étude est recalculé avec le même exécutable,
les mêmes réglages et le même protocole d'initialisation que les variantes.
Cela permet une comparaison interne claire, mais pas d'affirmer une reproduction
exacte des pourcentages historiques sans les scripts historiques correspondants.

## Mesures et erreurs

`pf.json.seconds` reste le temps mural interne PF, jamais divisé par les workers.
Le temps du processus entier et les heures UTC début/fin sont consignés séparément
dans `EXECUTION.json`. La fin UTC est celle du processus CLI, après sérialisation,
pas un horodatage prétendu exact du retour de `Solver::solve`.
`objective_evaluations = objective_pf + objective_local` est vérifié.
Les évaluations finales d'audit sont séparées dans le format natif PF131.

Le plafond de 1000 itérations n'impose pas des budgets d'évaluations égaux aux
variantes sans recherche locale. Les tableaux ajoutent donc temps et évaluations ;
il ne faut pas transformer leurs pourcentages en comparaison à coût identique.
Un arrêt budget est un résultat terminal valide. Une erreur du programme, un JSON
incomplet ou une option non appliquée fait arrêter le lanceur et n'est pas transformé
en échec scientifique. Une reprise conserve les tentatives incomplètes et utilise un
nouveau dossier `attempt_...` seulement pour ces tentatives, sans écraser les résultats.

## Référence

Fichier utilisateur : PFO_LJ_JOGO_SN_R2(1).pdf, sections 4.2–4.3, tableaux 7–10,
pp. 20–21 ; paramètres généraux, tableau 1, p.16.
