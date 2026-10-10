# Conserver les améliorations indépendamment de la convergence

Pour une minimisation, un candidat local est utile s'il est admissible,
si son objectif évalué est fini et s'il améliore strictement l'objectif.
Le statut de convergence n'intervient pas dans cette sélection.

La version 1.3.1 maintient un meilleur candidat propre à chaque raffinement,
au niveau commun aux backends. Il doit respecter les bornes originales,
les bornes locales et les contraintes locales applicables (boules atomiques
pour Newton LJ/Ipopt lorsqu'elles sont activées). L'évaluation de l'objectif
est déjà comptée dans le budget ; la mémorisation ne réévalue pas le point.

- À la fin d'un appel local, le meilleur candidat admissible est renvoyé s'il
  améliore le point terminal. Cela vaut aussi pour une limite d'itérations ou
  un échec de recherche linéaire. Une particule peut donc progresser sans
  que cet appel ait convergé.
- Un candidat évalué mais rejeté par le mécanisme d'acceptation des pas peut
  être conservé. Cela ne modifie pas les règles internes d'acceptation des pas.
- En cas de limite de temps ou d'évaluations, le meilleur point global déjà
  évalué reste disponible et est renvoyé par le solveur global. Les résultats
  des workers sont fusionnés avant la propagation de l'arrêt. Chaque particule
  interrompue ayant un candidat admissible récupère ce candidat et son objectif
  mémorisé, y compris dans le chemin séquentiel.
- `local_improvements` compte aussi les appels interrompus ayant trouvé une
  amélioration stricte par rapport à leur objectif initial.
- `local_converged`, `local_limited` et `local_failed` conservent le statut de
  l'appel local. Ils ne constituent pas un certificat de stationnarité du
  candidat retenu : celui-ci peut provenir d'un essai antérieur. L'audit de
  `pf_local_bench` vérifie séparément le point effectivement renvoyé.

Le contexte de chaque worker possède son propre candidat. Aucun état mutable
de candidat n'est partagé entre raffinements parallèles. Les erreurs fatales
de configuration ou de bibliothèque restent signalées comme telles.

## Régressions

`ctest --test-dir build_release -R local_incumbent --output-on-failure`
vérifie sur LJ2, avec une seule itération puis avec un budget d'évaluations
insuffisant pour converger, que les points conservés améliorent l'objectif,
restent admissibles et ont une valeur cohérente avec leurs coordonnées.
Les chemins CG, LDL, METIS+LDL, MUMPS, METIS+MUMPS et Ipopt/MUMPS sont testés
selon les dépendances disponibles, en série et avec deux workers.

Un objectif analytique oscillant très mal conditionné force en plus un échec
Armijo après évaluation d'un point améliorant. Le test vérifie le retour de
ce point et la conservation du statut d'échec, ainsi que la récupération
après interruption pendant la recherche linéaire.

Le meilleur point global était déjà mémorisé en 1.3.0 sans exiger la convergence.
Le renforcement 1.3.1 porte sur le retour commun par particule et le comptage
des améliorations interrompues. Il peut modifier une trajectoire PF par rapport
à la 1.3.0, sans modifier les graines ni les populations initiales.
