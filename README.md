# DataAnalysis

Adatbányászat és gépi tanulás beadandó feladat
A szívbetegségek előrejelzése adatelemzési eszközökkel
Vémann Sára 

Tartalomjegyzék
1.	Bevezetés, adatok értelmezése	3
2.	A modell	7
2.1.	Logisztikus Regresszó	7
2.2.	K-nearest neighbours (KNN)	9
2.3.	Support Vektor Machine (SVM)	10
3.	Összegzés	11



















Bevezetés, adatok értelmezése
A szív- és érrendszeri betegségek világszerte vezető haláloknak számítanak, évente milliók életét követelve. Ezen betegségek korai felismerése és hatékony kezelése kulcsfontosságú a halálozási arány csökkentése érdekében. Az orvostudomány fejlődésével és a rendelkezésre álló adatok mennyiségének növekedésével az adatelemzés egyre fontosabb szerepet játszik a szívbetegségek diagnosztizálásában és kezelésében.
A dolgozat célja, hogy készítsek egy olyan modellt, ami azonosítani tudja a szívbetegségeknek kitett egyéneket.
A rendelkezésemre álló adatokat a Magyar Kardiológiai Intézet 1988-as adatbázisa tartalmazza, amelyben 14 attribútum található:
	Kor (age): Az életkor
	Nem (sex): Nem (1 = férfi, 0 = nő).
	Mellkasi fájdalom típusa (cp):
	1: Tipikus angina
	2: Atipikus angina
	3: Nem anginás fájdalom
	4: Tünetmentes
	Nyugalmi vérnyomás (trestbps): Nyugalmi vérnyomás (mm Hg).
	Koleszterinszint (chol): Koleszterinszint (mg/dl).
	Éhgyomri vércukor (fbs): (Éhgyomri vércukor > 120 mg/dl) (1 = igaz, 0 = hamis).
	Nyugalmi EKG eredmények (restecg):
	0: Normál
	1: ST-T hullám abnormalitás (T hullám inverziók és/vagy ST eleváció vagy depresszió > 0.05 mV)
	2: Valószínű vagy határozott bal kamrai hipertrofia
	Maximális elért pulzus (thalach): Maximális elért pulzus.
	Terhelés indukálta angina (exang): Terhelés indukálta angina (1 = igen, 0 = nem).
	ST depresszió (oldpeak): ST depresszió terhelés hatására.
	ST szegmens lejtése (slope):
	1: Emelkedő
	2: Lapos
	3: Csökkenő
	Főbb erek száma (ca): Főbb erek száma (0-3), amelyek fluoreszkópiával láthatók.
	Thalassemia (thal):
	3: Normál
	6: Fix defektus
	7: Reverzibilis defektus
	Diagnózis (num): Szívbetegség diagnózisa (célváltozó).
	0: < 50% átmérőszűkület
	1: > 50% átmérőszűkület (1,2,3,4-es érték)
Az adatok elemzése során a bináris klasszifikáció megközelítést választottam. Ez a módszer különösen alkalmas a jelenlegi adathalmazra, mivel a célváltozó (szívbetegség diagnózisa) két lehetséges kimenetelre szűkíthető: a betegség jelenlétére (értékek 1-4) és annak hiányára (érték 0).
Az elemzést az adatok feldolgozásával kezdtem, két nagy kategóriát különböztethetünk meg az egyik a numerikus a másik, pedig a kategorikus változó. A numerikus változók számmal kifejezhetőek és mérhetőek, az alábbi attribútumokat soroltam ide: 
	Kor (age)
	Nyugalmi vérnyomás (trestbps)
	Koleszterinszint (chol)
	Maximális elért pulzus (thalach)
	ST depresszió (oldpeak)
	Főbb erek száma (ca)
A kategórikus változók pedig a következők:
	Nem (sex) 
	Mellkasi fájdalom típusa (cp)
	Éhgyomri vércukor (fbs)
	Nyugalmi EKG eredmények (restecg)
	Terhelés indukálta angina (exang)
	ST szegmens lejtése (slope)
	Thalassemia (thal)

Az adatok kategórikus változóinak kezelésére a "one-hot encoding" módszert választottam, amely minden kategorikus értéket bináris vektorokká alakít, hogy azokat könnyen kezelni lehessen a gépi tanulási modellekben. Azért erre esett a választásom, mivel a kategorikus változóim vagy bináris értéket vesznek fel, vagy több kategóriába tartozhatnak, amelyek között nincs sorrendiség.
A numerikus változók közül a főbb erek számánál (ca), az adatok több mint 90%- hiányzó érték volt, ezért eldobtam ezt a változót.
Ezek után egy korrelációs mátrix segítségével megvizsgáltam az adatokat.
 
1. ábra Korrelációs mátrix
Az mátrix segítségével jól látható, hogy számos magyarázó változó között erős, 0,5-nél nagyobb korreláció figyelhető meg. Ez problémát jelent, mivel a nagy korrelációk multikollinearitáshoz vezethetnek, ez azt jelenti, hogy a magyarázó változók együtt mozognak és ugyanazt a mintázatot magyarázzák a modelben. Erős negatív korrelációt figyelhetünk meg többek között az age (kor) és a max_heart_rate (maximális pulzusszám között), vagy erős pozitív korrelációt az ST_slope_flat (ST görbe lapos lejtése) és a ST_depression (ST depresszió) között
 Ezt követően vizsgáltam a VIF (Variance Inflation Factor) mutatókat annak érdekében, hogy meggyőződjek a multikollinearitás jelenlétéről. Ez a szám megmutatja, hogy mekkora összefüggés van a magyarázó változóink között.
VIF = 1/(1-R^2 )
Itt az R2 azt mutatja meg, hogy a magyarázó változó varianciája mennyire magyarázható a többi változóval szemben. Ha VIF értéke egyenlő lesz 1-gyel, akkor nincs multikollinearitás a modelben, ha 1<VIF≤5, akkor elfogadható, ha 5<VIF≤10 között van akkor nagy a multikollinearitás, a ha pedig 10-nél nagyobb akkor az szignifikáns multikollinearitásra utal. Mivel az én modellemben az age, a resting_blood_pressure és a max_heart_rate is nagyon magas értéket mutatott, ennek a problémának az orvoslására a PCA (Principal Component Analysis) dimenzó csökkentő eljárást alkalmaztam. Ez az eljárás új változókat hoz létre, amelyek felírhatóak az eredeti változók lineáris kombinációjaként, viszont ezek a létrehozott fő komponensek függetlennek lesznek egymástól, miközben az adatok varianciájának minél nagyobb részét igyekeznek megőrizni. Az így létrehozott adathalmaz dimenzionalitása kisebb lesz és ahogy az alábbi ábrán láthatjuk a VIF értékek alapján, így már a multikollinearitás sem áll fenn. Fontos megjegyezni, hogy mivel a PCA érzékeny az adatok skálájára, ezért előtte standardizáltam az adatokat, hogy 0 és 1 közötti értéket vegyenek fel. A komponensek számát úgy határoztam meg, hogy az adatok varianciájának 95%-a megmaradjon, így 15 új komponenst kaptam.
  
2. ábra VIF mutatók a PCA előtt és után
A következő lépésben megvizsgáltam a célváltozó (heart_desease_yes) azaz, a szívbetegség jelenlétének osztályeloszlását. A lenti ábrán láthatjuk, hogy kiegyensúlyozatlan az adathalmaz, ezért az „under sampling” módszert alkalmazva, a többségi osztályt (ami a nem szívbeteg kategória) leredukáltam a kisebbségi osztály elemszámára, így már kiegyensúlyozott osztályaim lettek. Azért választottam ezt a módszert, mert nem volt lehetőségem új valós adatok bevonására. Az azonos osztályeloszlás azért fontos, hogy a modell mindkét csoport karakterisztikáit egyenlő mértékben meg tudja tanulni.
 
3. ábra Osztályeloszlás, a többségi osztály csökkentése előtt
A modell
A modellek létrehozása előtt annak érdekében, hogy ne tanuljon rá túlságosan az adathalmaz sepcifikumaira a „train-test split” eljárást alkalmazva, szétszedtem az adatokat 20-80% arányban, hogy az adatok 20%-val tudjam tesztelni modellemet. Ez nagyon fontos az „overfitting”, tehát az adathalmazra való túlzott rátanulás miatt, mivel, ha ez bekövetkezne, akkor elveszítené a modell az általánosító képességét az új adatokkal szemben.
Logisztikus Regresszó
A logisztikus regresszió modelljének használatakor fontos, hogy optimalizáljuk a hiperparamétereket annak érdekében, hogy a modell a lehető legjobb teljesítményt nyújtsa. Ehhez a „GridSearch” eljárást alkalmaztam, ami egy adott listából választja ki a legoptimálisabb hiperparamétereket.
Az alábbi hiperparamétereket állítottam:
    penalty = ['l1', 'l2']
    C = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    class_weight = [{1:0.5, 0:0.5}, {1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.8, 0:0.2}, {1:0.7, 0: 0.3}, {1:0.6, 0:0.4}]
    solver = ['liblinear', 'saga']	

A gridSearchCV alapján pedig az alébbi hiperparaméterekkel lett a modell a legoptimálisabb:
 
Ezek után lefuttattam a modellt és a teszt adatokon végeztem el a kiértékelését.
log_model = train_logistic_regression(X_train_pca, y_train, X_test_pca, y_test,**best_params)

A modell pontossága (Accuracy) 84,75% ami azt mutatja, hogy a modell képes a legtöbb esetben helyesen azonosítani a szívbetegséget. Ez a magas pontosság arra utal, hogy a modell megbízható és jól használható a gyakorlatban.
Az AUC érték (AUC score) 0.931 azt jelzi, hogy a modell kiválóan teljesít a pozitív és negatív osztályok megkülönböztetésében. Tehát a modell jól tudja különválasztani a szívbetegeket és az egészségeseket.
 
A kapott eredményeket a legjobban hibamátrixal lehet szemléltetni. A mátrix négy kategóriába sorolja az adatokat. A mi esetünkben a TN (true negative) kategóriába 35 db adat került, ez azt jelenti, hogy a modellünk 35 esetben jól azonosította, hogy ha valaki nem szívbeteg. Ezzel szemben a TP (true positive) kategóriában a modell 15 esetben állapította meg jól szívbetegség jelenléltét. Az FP (false positive) kategóriába 3 adat került, tehát a modell 3 embert tévedésből a szívbeteg kategóriába sorolt, amikor nem azok. Ezzel szemben viszont a FN (false negative) esetben 6 embert sorolt, a nem szívbeteg kategóriába, akik szívbeteg.
Az F1 pontszám a pontosság és a visszahívás harmonikus átlaga. 
F1=2*  (P*R)/(P+R)
Itt a pontosság (P-precision), azt mondja meg, hogy a modell a pozitív osztályba sorolt adatai közül mennyi volt ténylegesen pozitív, a visszahívás (R-recall), pedig azt mutatja, hogy az összes pozitív esetből hányat sikerült ténylegesen azonosítani. Az én modellemben az F1 pontszám 0.769, ami átlagosnak mondható és azt a fejezi ki, hogy mennyire hatékonyan találja meg a pozitív eseteket és mennyire pontosak ezeknek az eseteknek az előrejelzése. 
 
4. ábra Hibamátrix

Véleményem szerint, egy orvosi döntéshozatalnál nagyobb probléma, ha valakit nem diagnosztizálnak betegnek, amikor az, ezért megvizsgálom, hogy a maradék két modellem, hogy teljesít.
K-nearest neighbours (KNN)
A KNN algoritmus a legfontosabb hiperparaméterének beállításával kezdtem, ami a K (n_neighbours), ez meghatározza, hogy hány közelebbi szomszédot vegyen figyelembe a döntéshozatal során. Ez azért fontos, mivel túl kevés szomszéd esetén érzékenyebb lesz a zajra, ami túlilleszkedést (overfitting) eredményez, túl sok szomszéd, pedig alulillesztéshez (underfitting) vezethet.
   
Az eredmények alapján a 18 szomszéd számot választottam, mivel az AUC itt volt a legmagasabb 0.93, egy darabig stagnált, majd 25 szomszédnál elkezdett csökkeni. A választásnál arra törekedtem, hogy egy olyan modellem legyen, ami a lehető legjobban különbözteti meg a szívbetegeket a nem szívbetegektől.
 
5. ábra Hibamátrix KNN
A KNN modell pontossága 83% az F1 értéke pedig 0,72. Az adatokból láthatjuk, hogy ez a modell kicsit pontatlanabb és összeségében több embert sorol a nem szívbeteg kategóriába, 8 esetben tévesen, szemben a logisztokus regresszióval, ahol ez 6 eset volt.
Support Vektor Machine (SVM)
Az SVM algoritmust is az optimális hiperparaméterek megkeresésével kezdtem.
best_svm = svm_hyperparameter_tuning(X_train, y_train, 
kernel_options=['linear', 'rbf', 'poly'],
c=[0.1, 1.0, 10.0, 15.0, 20.0, 25.0])

A GridSearchCV alapján a legjobb paraméter a C=15 és kernel=’linear’ lett. Ezekkel a hiperparaméterekkel az AUC 0.90 lett, a pontosság 83% és az F1 értéke 0.74 lett.
Az AUC ebben a modellben lett a legalacsonyabb, de még így is magasnak számít, tehát jól meg tudja különböztetni a modell a szívbetegeket a nem szívbetegektől.
 
6. ábra Hibamátrix SVM
A modell pontossága is hasonló az előzőekhez, habár, ha összehasonlítjuk a modellt a KNN-nel, akkor láthatjuk, hogy egy kicsivel inkább több beteget sorol a szívbeteg kategóriába, ami orvosi előrejelzéseknél szerintem fontos, mivel így több mindenkiről ki tud derülni a betegség jelenléte, és kevesebb embert sorolunk véletlenül az egészséges kategóriába.
 Összegzés
A dolgozatban három modellt készítettem és értékeltem ki, annak érdekében, hogy minél jobb eredménnyel meg tudjam állapítani a szívbetegség jelenlétét. Az alábbi táblázat segít a modellek összehasonlításában.
	Accuracy	AUC	F1	Hibák
Logisztikus regresszió	84,75%	0,93	0,76	6 FN
3 FP
KNN	83%	0,93	0,72	8 FN
2 FP
SVM	83%	0,90	0,74	7 FN
3 FP

Az eredmények alapján a logisztikus regresszió modellje bizonyult a legjobbnak mind a pontosság, mind az AUC és az F1 pontszám tekintetében. Ez a modell képes a legtöbb esetben helyesen azonosítani a szívbetegséget, és jól megkülönbözteti a pozitív és negatív osztályokat. Az orvosi döntéshozatal során különösen fontos, hogy minimalizáljuk a false negative esetek számát, azaz a nem diagnosztizált szívbetegek számát.
Azonban a többi modell sem rossz, mind a KNN és at SWM is 0,9 feletti AUC pontszámmal rendelkezik, tehát az esetek több mint 90%-ban jól meg tudják különböztetni egymástól a két osztályt.
