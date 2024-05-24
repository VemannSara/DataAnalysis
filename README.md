
<a name="_toc49137848"></a><a name="_toc49143739"></a><a name="_toc49919691"></a><a name="_toc116895324"></a><a name="_toc118117001"></a><a name="_toc149456468"></a><a name="_toc165971860"></a>

# Adatbányászat és gépi tanulás beadandó feladat

# A szívbetegségek előrejelzése adatelemzési eszközökkel

**Vémann Sára**

#
**Tartalomjegyzék**

[1.**	**BEVEZETÉS, ADATOK ÉRTELMEZÉSE	3****](#_toc166938217)**

[**2.**	**A MODELL	7****](#_toc166938218)

[2.1.	Logisztikus Regresszó	7](#_toc166938219)

[2.2.	K-nearest neighbours (KNN)	9](#_toc166938220)

[2.3.	Support Vektor Machine (SVM)	10](#_toc166938221)

[**3.**	**ÖSSZEGZÉS	11****](#_toc166938222)




















1. # <a name="_toc166938217"></a>**Bevezetés, adatok értelmezése**
A szív- és érrendszeri betegségek világszerte vezető haláloknak számítanak, évente milliók életét követelve. Ezen betegségek korai felismerése és hatékony kezelése kulcsfontosságú a halálozási arány csökkentése érdekében. Az orvostudomány fejlődésével és a rendelkezésre álló adatok mennyiségének növekedésével az adatelemzés egyre fontosabb szerepet játszik a szívbetegségek diagnosztizálásában és kezelésében.

A dolgozat célja, hogy készítsek egy olyan modellt, ami azonosítani tudja a szívbetegségeknek kitett egyéneket.

A rendelkezésemre álló adatokat a Magyar Kardiológiai Intézet 1988-as adatbázisa tartalmazza, amelyben 14 attribútum található:

1. **Kor (age)**: Az életkor
1. **Nem (sex)**: Nem (1 = férfi, 0 = nő).
1. **Mellkasi fájdalom típusa (cp)**:
   1. 1: Tipikus angina
   1. 2: Atipikus angina
   1. 3: Nem anginás fájdalom
   1. 4: Tünetmentes
1. **Nyugalmi vérnyomás (trestbps)**: Nyugalmi vérnyomás (mm Hg).
1. **Koleszterinszint (chol)**: Koleszterinszint (mg/dl).
1. **Éhgyomri vércukor (fbs)**: (Éhgyomri vércukor > 120 mg/dl) (1 = igaz, 0 = hamis).
1. **Nyugalmi EKG eredmények (restecg)**:
   1. 0: Normál
   1. 1: ST-T hullám abnormalitás (T hullám inverziók és/vagy ST eleváció vagy depresszió > 0.05 mV)
   1. 2: Valószínű vagy határozott bal kamrai hipertrofia
1. **Maximális elért pulzus (thalach)**: Maximális elért pulzus.
1. **Terhelés indukálta angina (exang)**: Terhelés indukálta angina (1 = igen, 0 = nem).
1. **ST depresszió (oldpeak)**: ST depresszió terhelés hatására.
1. **ST szegmens lejtése (slope)**:
   1. 1: Emelkedő
   1. 2: Lapos
   1. 3: Csökkenő
1. **Főbb erek száma (ca)**: Főbb erek száma (0-3), amelyek fluoreszkópiával láthatók.
1. **Thalassemia (thal)**:
   1. 3: Normál
   1. 6: Fix defektus
   1. 7: Reverzibilis defektus
1. **Diagnózis (num)**: Szívbetegség diagnózisa (célváltozó).
   1. 0: < 50% átmérőszűkület
   1. 1: > 50% átmérőszűkület (1,2,3,4-es érték)

Az adatok elemzése során a bináris klasszifikáció megközelítést választottam. Ez a módszer különösen alkalmas a jelenlegi adathalmazra, mivel a célváltozó (szívbetegség diagnózisa) két lehetséges kimenetelre szűkíthető: a betegség jelenlétére (értékek 1-4) és annak hiányára (érték 0).

Az elemzést az adatok feldolgozásával kezdtem, két nagy kategóriát különböztethetünk meg az egyik a numerikus a másik, pedig a kategorikus változó. A numerikus változók számmal kifejezhetőek és mérhetőek, az alábbi attribútumokat soroltam ide: 

- **Kor (age)**
- **Nyugalmi vérnyomás (trestbps)**
- **Koleszterinszint (chol)**
- **Maximális elért pulzus (thalach)**
- **ST depresszió (oldpeak)**
- **Főbb erek száma (ca)**

A kategórikus változók pedig a következők:

- **Nem (sex)** 
- **Mellkasi fájdalom típusa (cp)**
- **Éhgyomri vércukor (fbs)**
- **Nyugalmi EKG eredmények (restecg)**
- **Terhelés indukálta angina (exang)**
- **ST szegmens lejtése (slope)**
- **Thalassemia (thal)**


Az adatok kategórikus változóinak kezelésére a "one-hot encoding" módszert választottam, amely minden kategorikus értéket bináris vektorokká alakít, hogy azokat könnyen kezelni lehessen a gépi tanulási modellekben. Azért erre esett a választásom, mivel a kategorikus változóim vagy bináris értéket vesznek fel, vagy több kategóriába tartozhatnak, amelyek között nincs sorrendiség.

A numerikus változók közül a főbb erek számánál (ca), az adatok több mint 90%- hiányzó érték volt, ezért eldobtam ezt a változót.

Ezek után egy korrelációs mátrix segítségével megvizsgáltam az adatokat.

![image](https://github.com/VemannSara/DataAnalysis/assets/131291055/f0fe508c-b9bd-4996-a464-921e3d9fdd27)


1\*. ábra Korrelációs mátrix*

Az mátrix segítségével jól látható, hogy számos magyarázó változó között erős, 0,5-nél nagyobb korreláció figyelhető meg. Ez problémát jelent, mivel a nagy korrelációk multikollinearitáshoz vezethetnek, ez azt jelenti, hogy a magyarázó változók együtt mozognak és ugyanazt a mintázatot magyarázzák a modelben. Erős negatív korrelációt figyelhetünk meg többek között az age (kor) és a max\_heart\_rate (maximális pulzusszám között), vagy erős pozitív korrelációt az ST\_slope\_flat (ST görbe lapos lejtése) és a ST\_depression (ST depresszió) között

` `Ezt követően vizsgáltam a VIF (Variance Inflation Factor) mutatókat annak érdekében, hogy meggyőződjek a multikollinearitás jelenlétéről. Ez a szám megmutatja, hogy mekkora összefüggés van a magyarázó változóink között.

VIF = 1/(1-R2)

Itt az R<sup>2</sup> azt mutatja meg, hogy a magyarázó változó varianciája mennyire magyarázható a többi változóval szemben. Ha VIF értéke egyenlő lesz 1-gyel, akkor nincs multikollinearitás a modelben, ha 1<VIF≤5, akkor elfogadható, ha 5<VIF≤10 között van akkor nagy a multikollinearitás, a ha pedig 10-nél nagyobb akkor az szignifikáns multikollinearitásra utal. Mivel az én modellemben az age, a resting\_blood\_pressure és a max\_heart\_rate is nagyon magas értéket mutatott, ennek a problémának az orvoslására a PCA (Principal Component Analysis) dimenzó csökkentő eljárást alkalmaztam. Ez az eljárás új változókat hoz létre, amelyek felírhatóak az eredeti változók lineáris kombinációjaként, viszont ezek a létrehozott fő komponensek függetlennek lesznek egymástól, miközben az adatok varianciájának minél nagyobb részét igyekeznek megőrizni. Az így létrehozott adathalmaz dimenzionalitása kisebb lesz és ahogy az alábbi ábrán láthatjuk a VIF értékek alapján, így már a multikollinearitás sem áll fenn. Fontos megjegyezni, hogy mivel a PCA érzékeny az adatok skálájára, ezért előtte standardizáltam az adatokat, hogy 0 és 1 közötti értéket vegyenek fel. A komponensek számát úgy határoztam meg, hogy az adatok varianciájának 95%-a megmaradjon, így 15 új komponenst kaptam.

![image](https://github.com/VemannSara/DataAnalysis/assets/131291055/871f9c51-3f26-4ab1-ac4e-4ef6f3bf6b67)
![image](https://github.com/VemannSara/DataAnalysis/assets/131291055/a165615d-6419-4aaf-8ed5-fbacf4a94048)


2\*. ábra VIF mutatók a PCA előtt és után*

A következő lépésben megvizsgáltam a célváltozó (heart\_desease\_yes) azaz, a szívbetegség jelenlétének osztályeloszlását. A lenti ábrán láthatjuk, hogy kiegyensúlyozatlan az adathalmaz, ezért az „under sampling” módszert alkalmazva, a többségi osztályt (ami a nem szívbeteg kategória) leredukáltam a kisebbségi osztály elemszámára, így már kiegyensúlyozott osztályaim lettek. Azért választottam ezt a módszert, mert nem volt lehetőségem új valós adatok bevonására. Az azonos osztályeloszlás azért fontos, hogy a modell mindkét csoport karakterisztikáit egyenlő mértékben meg tudja tanulni.

![image](https://github.com/VemannSara/DataAnalysis/assets/131291055/902d3867-e25a-45bd-a8e6-18ff4bd17454)



3\*. ábra Osztályeloszlás, a többségi osztály csökkentése előtt*
1. # <a name="_toc166938218"></a>**A modell**
A modellek létrehozása előtt annak érdekében, hogy ne tanuljon rá túlságosan az adathalmaz sepcifikumaira a „train-test split” eljárást alkalmazva, szétszedtem az adatokat 20-80% arányban, hogy az adatok 20%-val tudjam tesztelni modellemet. Ez nagyon fontos az „overfitting”, tehát az adathalmazra való túlzott rátanulás miatt, mivel, ha ez bekövetkezne, akkor elveszítené a modell az általánosító képességét az új adatokkal szemben.
1. ## <a name="_toc166938219"></a>**Logisztikus Regresszó**
A logisztikus regresszió modelljének használatakor fontos, hogy optimalizáljuk a hiperparamétereket annak érdekében, hogy a modell a lehető legjobb teljesítményt nyújtsa. Ehhez a „GridSearch” eljárást alkalmaztam, ami egy adott listából választja ki a legoptimálisabb hiperparamétereket.

Az alábbi hiperparamétereket állítottam:

`    `penalty = ['l1', 'l2']

`    `C = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

`    `class\_weight = [{1:0.5, 0:0.5}, {1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.8, 0:0.2}, {1:0.7, 0: 0.3}, {1:0.6, 0:0.4}]

`    `solver = ['liblinear', 'saga']	

A gridSearchCV alapján pedig az alébbi hiperparaméterekkel lett a modell a legoptimálisabb:

![image](https://github.com/VemannSara/DataAnalysis/assets/131291055/b2d65c0c-1746-4926-86e0-840054456b82)
)

Ezek után lefuttattam a modellt és a teszt adatokon végeztem el a kiértékelését.

log\_model = train\_logistic\_regression(X\_train\_pca, y\_train, X\_test\_pca, y\_test,\*\*best\_params)

A modell pontossága (Accuracy) 84,75% ami azt mutatja, hogy a modell képes a legtöbb esetben helyesen azonosítani a szívbetegséget. Ez a magas pontosság arra utal, hogy a modell megbízható és jól használható a gyakorlatban.

Az AUC érték (AUC score) 0.931 azt jelzi, hogy a modell kiválóan teljesít a pozitív és negatív osztályok megkülönböztetésében. Tehát a modell jól tudja különválasztani a szívbetegeket és az egészségeseket.

![image](https://github.com/VemannSara/DataAnalysis/assets/131291055/1f6f1c8b-e3d2-4f1b-bcba-fbf93265b10f)


A kapott eredményeket a legjobban hibamátrixal lehet szemléltetni. A mátrix négy kategóriába sorolja az adatokat. A mi esetünkben a TN (true negative) kategóriába 35 db adat került, ez azt jelenti, hogy a modellünk 35 esetben jól azonosította, hogy ha valaki nem szívbeteg. Ezzel szemben a TP (true positive) kategóriában a modell 15 esetben állapította meg jól szívbetegség jelenléltét. Az FP (false positive) kategóriába 3 adat került, tehát a modell 3 embert tévedésből a szívbeteg kategóriába sorolt, amikor nem azok. Ezzel szemben viszont a FN (false negative) esetben 6 embert sorolt, a nem szívbeteg kategóriába, akik szívbeteg.

Az F1 pontszám a pontosság és a visszahívás harmonikus átlaga. 

F1=2\* P\*RP+R

Itt a pontosság (P-precision), azt mondja meg, hogy a modell a pozitív osztályba sorolt adatai közül mennyi volt ténylegesen pozitív, a visszahívás (R-recall), pedig azt mutatja, hogy az összes pozitív esetből hányat sikerült ténylegesen azonosítani. Az én modellemben az F1 pontszám 0.769, ami átlagosnak mondható és azt a fejezi ki, hogy mennyire hatékonyan találja meg a pozitív eseteket és mennyire pontosak ezeknek az eseteknek az előrejelzése. 


![image](https://github.com/VemannSara/DataAnalysis/assets/131291055/6e95beb5-0dba-4443-b581-bcab6ae4a246)


4\*. ábra Hibamátrix*

Véleményem szerint, egy orvosi döntéshozatalnál nagyobb probléma, ha valakit nem diagnosztizálnak betegnek, amikor az, ezért megvizsgálom, hogy a maradék két modellem, hogy teljesít.
1. ## <a name="_toc166938220"></a>**K-nearest neighbours (KNN)**
A KNN algoritmus a legfontosabb hiperparaméterének beállításával kezdtem, ami a K (n\_neighbours), ez meghatározza, hogy hány közelebbi szomszédot vegyen figyelembe a döntéshozatal során. Ez azért fontos, mivel túl kevés szomszéd esetén érzékenyebb lesz a zajra, ami túlilleszkedést (overfitting) eredményez, túl sok szomszéd, pedig alulillesztéshez (underfitting) vezethet.

![image](https://github.com/VemannSara/DataAnalysis/assets/131291055/97daa782-bb58-4888-840a-ff4bb6938c27)
![image](https://github.com/VemannSara/DataAnalysis/assets/131291055/e069b3a6-3294-4d65-8b25-5474f93f0909)


Az eredmények alapján a 18 szomszéd számot választottam, mivel az AUC itt volt a legmagasabb 0.93, egy darabig stagnált, majd 25 szomszédnál elkezdett csökkeni. A választásnál arra törekedtem, hogy egy olyan modellem legyen, ami a lehető legjobban különbözteti meg a szívbetegeket a nem szívbetegektől.

![image](https://github.com/VemannSara/DataAnalysis/assets/131291055/33d3ca2b-b1d3-4429-be5c-97e4817a0da7)


5\*. ábra Hibamátrix KNN*

A KNN modell pontossága 83% az F1 értéke pedig 0,72. Az adatokból láthatjuk, hogy ez a modell kicsit pontatlanabb és összeségében több embert sorol a nem szívbeteg kategóriába, 8 esetben tévesen, szemben a logisztokus regresszióval, ahol ez 6 eset volt.
1. ## <a name="_toc166938221"></a>**Support Vektor Machine (SVM)**
Az SVM algoritmust is az optimális hiperparaméterek megkeresésével kezdtem.

best\_svm = svm\_hyperparameter\_tuning(X\_train, y\_train, 

kernel\_options=['linear', 'rbf', 'poly'],

c=[0.1, 1.0, 10.0, 15.0, 20.0, 25.0])

A GridSearchCV alapján a legjobb paraméter a C=15 és kernel=’linear’ lett. Ezekkel a hiperparaméterekkel az AUC 0.90 lett, a pontosság 83% és az F1 értéke 0.74 lett.

Az AUC ebben a modellben lett a legalacsonyabb, de még így is magasnak számít, tehát jól meg tudja különböztetni a modell a szívbetegeket a nem szívbetegektől.

![image](https://github.com/VemannSara/DataAnalysis/assets/131291055/259e1015-16ef-42f9-91ba-a4eed3abfcda)


6\*. ábra Hibamátrix SVM*

A modell pontossága is hasonló az előzőekhez, habár, ha összehasonlítjuk a modellt a KNN-nel, akkor láthatjuk, hogy egy kicsivel inkább több beteget sorol a szívbeteg kategóriába, ami orvosi előrejelzéseknél szerintem fontos, mivel így több mindenkiről ki tud derülni a betegség jelenléte, és kevesebb embert sorolunk véletlenül az egészséges kategóriába.
1. # ` `**<a name="_toc166938222"></a>Összegzés**
A dolgozatban három modellt készítettem és értékeltem ki, annak érdekében, hogy minél jobb eredménnyel meg tudjam állapítani a szívbetegség jelenlétét. Az alábbi táblázat segít a modellek összehasonlításában.

||**Accuracy**|**AUC**|**F1**|**Hibák**|
| - | :-: | :-: | :-: | :-: |
|**Logisztikus regresszió**|84,75%|0,93|0,76|<p>6 FN</p><p>3 FP</p>|
|**KNN**|83%|0,93|0,72|<p>8 FN</p><p>2 FP</p>|
|**SVM**|83%|0,90|0,74|<p>7 FN</p><p>3 FP</p>|

Az eredmények alapján a logisztikus regresszió modellje bizonyult a legjobbnak mind a pontosság, mind az AUC és az F1 pontszám tekintetében. Ez a modell képes a legtöbb esetben helyesen azonosítani a szívbetegséget, és jól megkülönbözteti a pozitív és negatív osztályokat. Az orvosi döntéshozatal során különösen fontos, hogy minimalizáljuk a false negative esetek számát, azaz a nem diagnosztizált szívbetegek számát.

Azonban a többi modell sem rossz, mind a KNN és at SWM is 0,9 feletti AUC pontszámmal rendelkezik, tehát az esetek több mint 90%-ban jól meg tudják különböztetni egymástól a két osztályt.





















