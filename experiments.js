const fs = require('fs');
const path = require('path');
// Import modulu pro vysoce přesné měření času, vestavěno přímo v Node.js
const { performance } = require('perf_hooks');
const {
    extractVocabulary,
    vectorizeObjects,
    calculateTfidfAndNorms,
    calculateUserProfile,
    recommend
} = require('./recommender');

/**
 * EXPERIMENT 1: Calculation Time vs Number of Objects
 * Simulates scaling in a Node.js environment.
 */
function runPerformanceExperiment(baseMovies) {
    console.log("--- EXPERIMENT 1: Calculation Time vs Number of Objects (Node.js) ---");
    // Definuje, s jakými velikostmi datasetu budeme zátěžový experiment provádět.
    const sizes = [20, 100, 500, 1000, 5000, 10000];
    
    sizes.forEach(size => {
        // Scale up the dataset by duplicating baseMovies
        // Proč duplikace dat (Load Testing simulace):
        // Protože naše původní databáze (movies.json) má omezený počet položek, uměle ji "nafukujeme".
        // Zacyklíme původní pole objektů a naklonujeme je s novým ID. To nám umožní exaktně změřit, jak 
        // se náš algoritmus pro matematickou transformaci zpomalí s obrovským objemem dat a prověřit tak
        // vliv jeho výpočetní složitosti O(n*m).
        const scaledMovies = [];
        for (let i = 0; i < size; i++) {
            // Používáme hluboké kopírování přes JSON, abychom zamezili referenčnímu provázání objektů
            const movie = JSON.parse(JSON.stringify(baseMovies[i % baseMovies.length]));
            movie.id = i + 1;
            movie.title = `${movie.title} (Copy ${i})`;
            scaledMovies.append ? null : scaledMovies.push(movie); 
        }
        
        // Zaznamenání času před začátkem výpočetně náročných operací
        // Proč performance.now(): Jedná se o sub-milisekundově přesnou metodu Node.js, 
        // která měří reálný uběhnutý čas běhu procesu na procesoru. Na rozdíl od běžného Date.now() 
        // není náchylná na změny systémového času, což garantuje naprostou exaktnost tohoto experimentu.
        const startTime = performance.now();
        
        // 1. Extraction (Extrakce slovníku všech žánrů z nafouknutého pole filmů)
        const vocab = extractVocabulary(scaledMovies);
        
        // 2. Vectorization (Převedení textových vlastností a čísel do surového pole/vektoru)
        const movieVectors = scaledMovies.map(m => vectorizeObjects(m, vocab));
        
        // 3. TF-IDF & Norms (Sestavení potřebných vah pro termy a Euklidovských délek k budoucí normalizaci)
        const { idfVector, norms } = calculateTfidfAndNorms(movieVectors, vocab.length);
        
        // 4. User Profile (Nasimulování implicitního ohodnocení třemi fiktivními filmy)
        const userRatings = { 0: 1, 1: 1, 8: -1 };
        const userProfile = calculateUserProfile(userRatings, movieVectors, norms);
        
        // 5. Recommendations (Samotné vynásobení matic pomocí skalárního součinu a seřazení výsledku)
        const recs = recommend(userProfile, idfVector, movieVectors, scaledMovies);
        
        // Změříme výsledný rozdíl času (delta)
        const endTime = performance.now();
        const calcTime = endTime - startTime;
        
        // Zde v konzoli typicky okolo 10 000 položek (při porovnání předchozích běhů) uvidíme tzv. "flatline". 
        // Čas přestane narůstat lineárně tak strmě, jelikož JIT (Just-In-Time) kompilátor enginu V8 rozpozná,
        // že se kritické smyčky opakují, a optimalizuje JavaScript přímo do superrychlého strojového kódu.
        console.log(`Dataset Size: ${size.toString().padStart(6)} objects | Calculation Time: ${calcTime.toFixed(2).padStart(7)} ms`);
    });
}

/**
 * EXPERIMENT 2: Vector Compilation Differences
 * Standard (Categories + Numeric) vs Categories ONLY.
 */
function runCompilationExperiment(movies) {
    console.log("\n--- EXPERIMENT 2: Vector Compilation Differences (Node.js) ---");
    
    // Base user ratings: likes Sci-Fi/Action (The Matrix), dislikes Drama (Titanic)
    // Proč tyto testovací vstupy: Záměrně uživateli nastavujeme protichůdné preference, abychom 
    // dokázali vyhodnotit citlivost algoritmu a otestovali různé přístupy k tvorbě vektorového prostoru.
    const userRatings = { 0: 1, 8: -1 }; 
    
    const vocab = extractVocabulary(movies);
    
    // --- 1. Standard Vectorization (Categories + Numeric) ---
    // Toto je produkční způsob popsaný v teorii. Mimo žánrů se do vektorů vkládá i hodnocení a rok.
    // Numerickým hodnotám je uvnitř přidělena IDF fixně na 1.0.
    const standardVectors = movies.map(m => vectorizeObjects(m, vocab));
    const { idfVector: idfStd, norms: normsStd } = calculateTfidfAndNorms(standardVectors, vocab.length);
    const profileStd = calculateUserProfile(userRatings, standardVectors, normsStd);
    const recsStd = recommend(profileStd, idfStd, standardVectors, movies);
    
    // --- 2. Alternative Vectorization (Categories ONLY) ---
    // Běžný způsob, kde by se Vektorový model tvořil absolutně jen podle žánrů. Tím vyřadíme z rovnice
    // numerická data, čímž vznikne obrovské riziko totožného skóre pro filmy, které mají jen stejný žánr.
    function vectorizeCategoriesOnly(movie, vocab) {
        const vector = [];
        const movieCategories = new Set(movie.categories || []);
        vocab.forEach(category => {
            vector.push(movieCategories.has(category) ? 1.0 : 0.0);
        });
        return vector;
    }
    
    const altVectors = movies.map(m => vectorizeCategoriesOnly(m, vocab));
    const { idfVector: idfAlt, norms: normsAlt } = calculateTfidfAndNorms(altVectors, vocab.length);
    const profileAlt = calculateUserProfile(userRatings, altVectors, normsAlt);
    const recsAlt = recommend(profileAlt, idfAlt, altVectors, movies);
    
    console.log("Comparing Top 3 Recommendations:");
    console.log(`${'Standard (Cats + Numeric)'.padEnd(40)} | ${'Alternative (Cats Only)'.padEnd(40)}`);
    console.log("-".repeat(85));
    
    // Proč se tento experiment provádí: Ve výstupu do konzole jasně prokazujeme, že čistě žánrový 
    // přístup (Alternative) vede k naprostým shodám v řazení doporučení (tie). Tím dokazujeme inženýrský
    // a teoretický přínos modelu zleva (Standard), kdy vložením numerických dimenzí s fixním IDF skóre 
    // "rozbíjíme shody" (tie-breaking) a získáváme tak jemnější a spravedlivější seřazení pro uživatele.
    for (let i = 0; i < 3; i++) {
        const stdTitle = `${recsStd[i].title} (${recsStd[i].score.toFixed(4)})`;
        const altTitle = `${recsAlt[i].title} (${recsAlt[i].score.toFixed(4)})`;
        console.log(`${stdTitle.padEnd(40)} | ${altTitle.padEnd(40)}`);
    }
}

// Execution
const moviesPath = path.join(__dirname, 'movies.json');
const movies = JSON.parse(fs.readFileSync(moviesPath, 'utf8'));

runPerformanceExperiment(movies);
runCompilationExperiment(movies);
