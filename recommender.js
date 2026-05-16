// recommender.js

function extractVocabulary(movies) {
    /**
     * Extrahuje seřazený seznam všech unikátních kategorií (žánrů) napříč všemi filmy.
     * Proč: K vytvoření Vektorového modelu potřebujeme znát dimenze našeho vektorového prostoru. 
     * Každý unikátní žánr představuje jednu dimenzi (osu) v tomto prostoru. Set (množina) je použit
     * pro zamezení duplicit.
     */
    const vocabSet = new Set();
    movies.forEach(movie => {
        const categories = movie.categories || [];
        categories.forEach(cat => vocabSet.add(cat));
    });
    return Array.from(vocabSet).sort();
}

function vectorizeObjects(movie, vocab) {
    /**
     * Převede konkrétní film a slovník (žánrů) do kombinovaného vektoru příznaků (feature vector).
     * Proč: Abychom mohli provádět matematické operace (např. měřit podobnost), 
     * musíme textové/vlastnostní popisy převést na čísla.
     */
    const vector = [];
    const movieCategories = new Set(movie.categories || []);
    
    // 1. Kategorické příznaky (1 nebo 0)
    // Proč: Procházíme celý slovník. Pokud film daný žánr má, přiřadíme 1.0 (TF - Term Frequency, zde je binární), 
    // jinak 0.0. To určuje pozici vektoru v dané dimenzi.
    vocab.forEach(category => {
        vector.push(movieCategories.has(category) ? 1.0 : 0.0);
    });
    
    // 2. Numerické příznaky
    // Proč: Vektor nemusí obsahovat jen žánry, ale i další vlastnosti jako rok vydání nebo hodnocení. 
    // Tyto vlastnosti přidávají další dimenze do našeho vektorového prostoru a pomáhají rozlišit filmy 
    // se stejnými žánry. Přidáváme je na konec vektoru.
    const normFeatures = movie.normalized_features || {};
    vector.push(normFeatures.year || 0.0);
    vector.push(normFeatures.rating || 0.0);
    
    return vector;
}

function calculateTfidfAndNorms(movieVectors, vocabLength) {
    /**
     * Počítá DF (Document Frequency), IDF (Inverse Document Frequency) a euklidovské normy (L2) všech vektorů.
     * Proč: TF-IDF je klíčový koncept Content-Based doporučování. Zvýhodňuje specifické, vzácné vlastnosti 
     * oproti těm obecným. Euklidovská norma zase zajišťuje, že všechny vektory budou mít srovnatelnou délku (délka 1).
     */
    const n = movieVectors.length;
    if (n === 0) return { idfVector: [], norms: [] };
    
    const numFeatures = movieVectors[0].length;
    
    // 1. Výpočet DF (Document Frequency)
    // Proč: Zjišťujeme, v kolika dokumentech (filmech) se vyskytuje daná vlastnost (žánr).
    const dfVector = new Array(numFeatures).fill(0);
    movieVectors.forEach(vector => {
        for (let k = 0; k < numFeatures; k++) {
            if (vector[k] > 0) dfVector[k]++;
        }
    });
    
    // 2. Výpočet IDF (Inverse Document Frequency) pomocí log10
    // Proč: IDF získáme vzorcem log(N / DF). Pokud je žánr u všech filmů, DF=N, a IDF vyjde 0.
    // Tato metoda .map transformuje pole DF na vektor vah podle vzorce pro IDF.
    const idfVector = dfVector.map((df, k) => {
        // Zásadní inženýrský "hack" / fix: 
        // Pokud je index k roven nebo větší než délka žánrů, jde o numerický příznak (rok, rating).
        // Tyto příznaky mají všechny filmy (tedy DF = N). Log(N/N) by vrátil 0 a numerický parametr by byl ignorován.
        // Proto numerickým příznakům fixně nastavujeme váhu 1.0, což nám pomáhá "rozbít shody" u filmů se stejnými žánry.
        if (k >= vocabLength) return 1.0;
        
        if (df === 0) return 0.0;
        return Math.log10(n / df);
    });
    
    // 3. Výpočet Norem (Euklidovská L2 norma)
    // Proč: Vypočítáme délku vektoru každého filmu. Vzorec je odmocnina ze součtu čtverců všech jeho prvků: sqrt(x1^2 + x2^2 + ...).
    // K tomu používáme .reduce(), které elegantně sečte čtverce hodnot. Pomocí těchto norem pak vektory vydělíme,
    // čímž srovnáme filmy s jedním žánrem a filmy se 4 žánry na stejnou výchozí pozici (jednotkovou délku).
    const norms = movieVectors.map(vector => {
        const sumSq = vector.reduce((acc, val) => acc + val * val, 0);
        return Math.sqrt(sumSq);
    });
    
    return { idfVector, norms };                   
}

function calculateUserProfile(userRatings, movieVectors, norms) {
    /**
     * Agreguje uživatelský profil (User Profile) na základě explicitních hodnocení.
     * Proč: Uživatelský profil je pomyslný vektor, který reprezentuje "těžiště" nebo ideální vkus uživatele.
     * Vypočítá se tak, že sečteme normalizované vektory filmů, které uživatel ohodnotil, a tyto vektory
     * ještě vynásobíme uživatelovým ratingem (např. +1 pro Líbí se, -1 pro Nelíbí se).
     */
    if (movieVectors.length === 0) return [];
    
    const numFeatures = movieVectors[0].length;
    const userProfile = new Array(numFeatures).fill(0.0);
    
    // Procházíme pouze filmy, které uživatel explicitně ohodnotil.
    Object.keys(userRatings).forEach(idxStr => {
        const idx = parseInt(idxStr, 10);
        const rating = userRatings[idx];
        
        if (rating !== 0) {
            const vector = movieVectors[idx];
            const norm = norms[idx];
            
            // Matematika: Pro každou dimenzi vektoru daného filmu (žánr, numerická hodnota)
            for (let k = 0; k < numFeatures; k++) {
                // Nejprve hodnotu vydělíme normou vektoru (čímž vytvoříme jednotkový vektor).
                const normalizedVal = norm > 0 ? vector[k] / norm : 0;
                // Následně ji vynásobíme hodnocením a přičteme do uživatelského profilu.
                userProfile[k] += normalizedVal * rating;
            }
        }
    });
    
    return userProfile;
}

function recommend(userProfile, idfVector, movieVectors, movies) {
    /**
     * Vypočítá finální skóre pro každý film v databázi a vrátí doporučení.
     * Proč: Zde dochází ke spárování vkusu uživatele (User Profile) s obsahem filmů za využití vah (IDF).
     */
    const numFeatures = userProfile.length;
    const recommendations = [];
    
    // Procházíme všechny filmy v databázi, i ty, které uživatel ještě neviděl (řešení problému Cold Start).
    movieVectors.forEach((vector, i) => {
        let score = 0.0;
        // Matematika: Skalární součin (Inner Product / Scalar Product) tří vektorů.
        // Pro každou dimenzi (k) vynásobíme: surovou hodnotu filmu * váhu IDF pro tuto dimenzi * hodnotu dimenze v uživatelském profilu.
        // Tím se projeví, nakolik daný film v dané vlastnosti rezonuje s vkusem uživatele a navíc se aplikuje váha IDF,
        // která upřednostní shody v raritních vlastnostech před těmi běžnými.
        for (let k = 0; k < numFeatures; k++) {
            score += vector[k] * idfVector[k] * userProfile[k];
        }
        recommendations.push({ score, title: movies[i].title });
    });
    
    // Seřazení výsledků sestupně. Film s nejvyšším skóre se uživateli líbí teoreticky nejvíce.
    recommendations.sort((a, b) => b.score - a.score);
    return recommendations;
}

module.exports = {
    extractVocabulary,
    vectorizeObjects,
    calculateTfidfAndNorms,
    calculateUserProfile,
    recommend
};
