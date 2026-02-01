## Exercice 1.1 — Mandelbrot avec répartition statique (par blocs de lignes)

Les temps d’exécution mesurés pour le calcul de l’ensemble de Mandelbrot sont les suivants :

| Nombre de processus (p) | Temps T(p) [s] | Speedup S(p) |
|------------------------:|---------------:|-------------:|
| 1 | 1.681 | 1.00 |
| 2 | 0.866 | 1.94 |
| 4 | 0.506 | 3.33 |
| 8 | 0.372 | 4.52 |

Le speedup est défini par :
\[
S(p) = \frac{T(1)}{T(p)}
\]

On observe que le temps d’exécution diminue lorsque le nombre de processus augmente, ce qui montre que la parallélisation est efficace.  
Cependant, le speedup reste sous-linéaire et inférieur au speedup idéal \(S(p)=p\).  
Cela s’explique par la présence d’une partie séquentielle (reconstruction et sauvegarde de l’image sur le processus maître), par les coûts de synchronisation et de communication MPI, ainsi que par le surcoût de gestion des processus.

Lorsque le nombre de processus devient élevé, la quantité de travail par processus diminue et l’overhead MPI devient prédominant, ce qui limite les gains de performance.


## Exercice 1.2 — Répartition statique cyclique (round-robin)

Les temps d’exécution mesurés avec une répartition cyclique des lignes sont les suivants :

| Nombre de processus (p) | Temps T(p) [s] | Speedup S(p) |
|------------------------:|---------------:|-------------:|
| 1 | 1.665 | 1.00 |
| 2 | 0.919 | 1.81 |
| 4 | 0.481 | 3.46 |
| 8 | 0.338 | 4.92 |

Le speedup est défini par :
\[
S(p) = \frac{T(1)}{T(p)}
\]

On observe que la répartition cyclique permet un meilleur équilibrage de charge par rapport à la répartition par blocs contigus (Ex 1.1), en particulier lorsque le nombre de processus augmente.  
Le speedup obtenu est légèrement supérieur à celui de l’Exercice 1.1, ce qui montre que le temps total est moins contraint par les processus les plus lents.

Cependant, cette stratégie présente aussi des limites : la reconstruction de l’image sur le processus maître est plus complexe et la localité mémoire est moins bonne que dans le cas de blocs contigus.  
La répartition reste statique, ce qui motive l’utilisation d’une stratégie dynamique (maître–esclave) à l’Exercice 1.3.


## Exercice 1.3 — Stratégie dynamique maître–esclave

Les temps d’exécution mesurés avec la stratégie maître–esclave sont les suivants
(le temps séquentiel \(T(1)\) est celui du code non parallèle) :

| Nombre de processus (p) | Temps T(p) [s] | Speedup S(p) |
|------------------------:|---------------:|-------------:|
| 2 | 1.905 | 0.88 |
| 4 | 0.635 | 2.65 |
| 8 | 0.359 | 4.69 |

Le speedup est défini par :
\[
S(p) = \frac{T(1)}{T(p)}
\]

On constate que pour un faible nombre de processus (\(p = 2\)), la stratégie maître–esclave est pénalisée par un surcoût de communication important, ce qui conduit à un speedup inférieur à 1.  
En revanche, lorsque le nombre de processus augmente, l’équilibrage dynamique de la charge devient efficace et permet d’obtenir de meilleures performances.

Cette stratégie est particulièrement adaptée lorsque le coût de calcul varie fortement entre les tâches, mais elle peut souffrir d’un goulot d’étranglement au niveau du processus maître si le nombre de processus devient très élevé.


## Exercice 2 — Produit matrice-vecteur \(v = A\,u\)

On reprend exactement la définition du code séquentiel fourni : :contentReference[oaicite:1]{index=1}  
- `dim` : dimension du problème  
- \(A_{ij} = (i+j)\bmod dim + 1\)  
- \(u_j = j+1\)  
- \(v = A\cdot u\)

On suppose que `dim` est divisible par le nombre de processus `nbp`. :contentReference[oaicite:2]{index=2}

D’après le cours, le speed-up est défini par : :contentReference[oaicite:3]{index=3}  
\[
S(n)=\frac{t_s}{t_p(n)}
\]

---

### 2.a Partitionnement par colonnes

#### Principe
On définit :
\[
N_{loc}=\frac{dim}{nbp}
\]
Chaque processus possède :
- un bloc de colonnes \(A^{(k)} \in \mathbb{R}^{dim\times N_{loc}}\),
- le bloc correspondant \(u^{(k)} \in \mathbb{R}^{N_{loc}}\).

Chaque processus calcule une contribution partielle :
\[
v^{(k)} = A^{(k)}u^{(k)} \in \mathbb{R}^{dim}
\]
Puis on obtient le résultat final par une somme globale :
\[
v=\sum_{k=1}^{nbp} v^{(k)}
\]
Implémentation MPI : `MPI_Allreduce(..., op=SUM)`.

#### Mesures (dim = 8000)
Temps mesurés :
- \(t_p(1)=0.652776\,s\)
- \(t_p(2)=0.299958\,s\)
- \(t_p(4)=0.243507\,s\)
- \(t_p(8)=0.245048\,s\)

| nbp (n) | Temps \(t_p(n)\) [s] | Speed-up \(S(n)=t_p(1)/t_p(n)\) |
|--------:|----------------------:|--------------------------------:|
| 1 | 0.652776 | 1.00 |
| 2 | 0.299958 | 2.18 |
| 4 | 0.243507 | 2.68 |
| 8 | 0.245048 | 2.66 |

---

### 2.b Partitionnement par lignes

#### Principe
On définit :
\[
N_{loc}=\frac{dim}{nbp}
\]
Chaque processus possède :
- un bloc de lignes \(A_{loc}\in\mathbb{R}^{N_{loc}\times dim}\),
- l’intégralité de \(u\in\mathbb{R}^{dim}\).

Chaque processus calcule :
\[
v_{loc}=A_{loc}\,u \in \mathbb{R}^{N_{loc}}
\]
Puis on reconstruit \(v\) complet dans tous les processus via :
`MPI_Allgather(v_loc)`.

#### Mesures (dim = 8000)
Temps mesurés :
- \(t_p(1)=0.631987\,s\)
- \(t_p(2)=0.339721\,s\)
- \(t_p(4)=0.278579\,s\)
- \(t_p(8)=0.280718\,s\)

| nbp (n) | Temps \(t_p(n)\) [s] | Speed-up \(S(n)=t_p(1)/t_p(n)\) |
|--------:|----------------------:|--------------------------------:|
| 1 | 0.631987 | 1.00 |
| 2 | 0.339721 | 1.86 |
| 4 | 0.278579 | 2.27 |
| 8 | 0.280718 | 2.25 |

---

### Discussion (comparaison lignes vs colonnes)

- Pour `dim = 8000`, on observe un speed-up significatif jusqu’à \(n=4\), puis une saturation (voire légère dégradation) à \(n=8\).
- Cela s’explique par les notions du cours : lorsque \(n\) augmente, la granularité (quantité de calcul par processus) diminue et les surcoûts de communication/synchronisation prennent une place plus importante. :contentReference[oaicite:4]{index=4}
- Dans ces mesures, le partitionnement par colonnes donne de meilleures performances globales (speed-up ≈ 2.68 à 4 processus) que le partitionnement par lignes (≈ 2.27 à 4 processus), mais les deux approches finissent par saturer pour \(n=8\).

Conclusion : le produit matrice–vecteur parallèle présente un speed-up limité par l’overhead MPI et la baisse de granularité quand le nombre de processus devient élevé, même si `dim` est grand.


## 3. Entraînement pour l’examen écrit  

On note :
- \(t_s\) : temps d’exécution séquentiel total ;
- \(f\) : fraction du temps séquentiel correspondant à la partie non parallélisable.

D’après l’énoncé :
\[
f = 0.1 \qquad (90\% \text{ du temps est parallélisable})
\]

---

### 3.1 Accélération maximale — Loi d’Amdahl (\(n \gg 1\))

D’après le cours, la loi d’Amdahl s’écrit :
\[
S(n) = \frac{t_s}{f\,t_s + \frac{(1-f)t_s}{n}}
     = \frac{n}{1 + (n-1)f}
\]

Quand \(n \to \infty\) :
\[
S_{\max} = \frac{1}{f}
\]

Avec \(f = 0.1\) :
\[
S_{\max} = \frac{1}{0.1} = \boxed{10}
\]

---

### 3.2 Nombre de nœuds raisonnable

La loi d’Amdahl montre que le speedup se sature rapidement lorsque \(f > 0\).
Au-delà d’un certain nombre de nœuds, l’augmentation de \(n\) apporte un gain marginal très faible.

Ainsi, pour \(f = 0.1\), il est raisonnable de limiter le nombre de nœuds à une valeur bien inférieure à l’infini (typiquement quelques dizaines), afin d’éviter de gaspiller des ressources CPU pour un gain de performance négligeable.

---

### 3.3 Accélération observée maximale : \(S \approx 4\)

Alice observe expérimentalement une accélération maximale d’environ 4, inférieure à la valeur théorique donnée par la loi d’Amdahl.

Cette différence s’explique par des phénomènes non pris en compte par le modèle idéal :
- coûts de communication ;
- synchronisations ;
- déséquilibre de charge ;
- surcoûts liés à l’architecture mémoire distribuée.

La fraction séquentielle effective est donc plus importante que la fraction théorique \(f = 0.1\).

---

### 3.4 Accélération maximale — Loi de Gustafson (données doublées)

D’après le cours, la loi de Gustafson repose sur les hypothèses suivantes :
- le temps séquentiel \(t_s\) est indépendant de la taille des données ;
- le temps parallèle \(t_p\) est linéaire avec la taille des données.

En normalisant :
\[
t_s + t_p = 1
\]

La loi de Gustafson s’écrit alors :
\[
S(n) = t_s + n\,t_p = n + (1-n)t_s
\]

Avec \(t_s = f = 0.1\) :
\[
S(n) = n + (1-n)\times 0.1 = 0.9n + 0.1
\]

Ainsi, lorsque la quantité de données est augmentée (par exemple doublée), la loi de Gustafson prédit une accélération quasi linéaire avec le nombre de nœuds, contrairement à la loi d’Amdahl.

Cela montre que, pour des problèmes de grande taille, l’utilisation d’un plus grand nombre de nœuds peut rester efficace.
