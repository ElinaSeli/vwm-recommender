const express = require('express');
const expressLayouts = require('express-ejs-layouts');
const fs = require('fs');
const path = require('path');
const {
    extractVocabulary,
    vectorizeObjects,
    calculateTfidfAndNorms,
    calculateUserProfile,
    recommend
} = require('./recommender');

const app = express();
const PORT = 3000;

// Setup EJS and Layouts
// Proč Node.js/Express: EJS (Embedded JavaScript) používáme jako šablonovací systém.
// Umožňuje nám to dynamicky generovat HTML na straně serveru a vkládat do něj proměnné, jako jsou naše doporučení.
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));
app.use(expressLayouts);
app.set('layout', 'base');
// Umožňuje Expressu parsovat data odeslaná přes formuláře (URL-encoded data z requestu).
app.use(express.urlencoded({ extended: true }));

// Load movies data globally from the parent directory
// Proč: Tímto jednorázově synchronně načteme obsah databáze při startu serveru,
// abychom nemuseli přistupovat na disk při každém požadavku na doporučení, čímž zásadně zrychlujeme zpracování.
const moviesData = fs.readFileSync(path.join(__dirname, 'movies.json'), 'utf-8');
const MOVIES = JSON.parse(moviesData);

// Global dummy database: { movie_index: rating }
// Proč in-memory: Pro prototypování ("Thin Server" architekturu) nepotřebujeme těžkou SQL/NoSQL databázi. 
// Objekt uložený v paměti funguje jako dočasná databáze. Zaznamenává hodnocení konkrétního uživatele.
// Pokud aplikaci restartujeme, stav se vynuluje. Výhodou je extrémně rychlé čtení a zápis, což 
// umožňuje backendu instantně pře-počítat matematické matice na základě změn.
const USER_RATINGS = {};

// Hlavní stránka (Route '/')
app.get('/', (req, res) => {
    // Proč res.render: Vygeneruje výsledné HTML ze šablony 'index' (ve složce views) a pošle ho jako odpověď (Response) klientovi.
    // Objekt, který posíláme jako druhý parametr { movies, user_ratings, ... }, představuje data (kontext),
    // do kterých mají EJS značky (např. <%= movie.title %>) přístup, a podle kterých se vygeneruje obsah stránky.
    res.render('index', { 
        movies: MOVIES, 
        user_ratings: USER_RATINGS,
        message: null
    });
});

// Endpoint pro hodnocení filmu (Route '/rate/:movie_id/:rating')
app.get('/rate/:movie_id/:rating', (req, res) => {
    // Proč req.params: req.params uchovává hodnoty dynamických částí z URL cesty (definovány dvojtečkou). 
    // Pokud je volána adresa "/rate/5/1", req.params.movie_id bude '5' a req.params.rating bude '1'.
    const movieId = parseInt(req.params.movie_id, 10);
    let rating = parseInt(req.params.rating, 10);
    
    if (isNaN(rating)) {
        return res.redirect('/');
    }
    
    // Pole indexujeme od nuly, id v URL od jedničky.
    const movieIndex = movieId - 1;
    
    // Zabezpečení proti neplatným hodnotám, simulace operace uložení/smazání v databázi.
    if (movieIndex >= 0 && movieIndex < MOVIES.length && [1, 0, -1].includes(rating)) {
        if (rating === 0) {
            // Hodnocení 0 znamená reset, takže klíč z naší dummy in-memory databáze zcela odstraníme.
            delete USER_RATINGS[movieIndex];
        } else {
            // Zapíšeme +1 nebo -1 k danému filmu do uživatelského profilu (objektu).
            USER_RATINGS[movieIndex] = rating;
        }
    }
    
    // Po zpracování požadavku přesměrujeme uživatele bezpečně zpět na domovskou stránku.
    res.redirect('/');
});

// Endpoint pro doporučení (Route '/recommendations')
app.get('/recommendations', (req, res) => {
    if (Object.keys(USER_RATINGS).length === 0) {
        return res.render('index', { 
            movies: MOVIES, 
            user_ratings: USER_RATINGS,
            message: { type: 'warning', text: 'You need to rate some movies first to get personalized recommendations!' }
        });
    }
    
    // Mathematics Logic Pipeline
    // Proč: Tady dochází k provedení kompletního doporučovacího řetězce "on-the-fly" v rámci jednoho HTTP požadavku.
    // Pro každý dotaz dynamicky vytvoříme Vektorový prostor nad naší in-memory databází, spočítáme TF-IDF a nakonec i skalární součin.
    const vocab = extractVocabulary(MOVIES);
    const movieVectors = MOVIES.map(m => vectorizeObjects(m, vocab));
    const { idfVector, norms } = calculateTfidfAndNorms(movieVectors, vocab.length);
    
    const userProfile = calculateUserProfile(USER_RATINGS, movieVectors, norms);
    const recs = recommend(userProfile, idfVector, movieVectors, MOVIES);
    
    // Vyrendruje a zašle klientovi šablonu 'recommendations' naplněnou hotovými výsledky skórování.
    res.render('recommendations', { 
        recommendations: recs,
        message: null
    });
});

// Endpoint pro zobrazení admin panelu (Route GET '/admin')
app.get('/admin', (req, res) => {
    res.render('admin', { message: null });
});

// Endpoint pro zpracování přepočtů z admin panelu (Route POST '/admin')
app.post('/admin', (req, res) => {
    // Proč simulace: V reálném produkčním nasazení by se zde spustil asynchronní "Cron" job na pozadí, který by např. v noci
    // prošel obrovskou relační databázi a přepočítal by předem TF-IDF váhy a normy pro miliony filmů. Zabránilo by se
    // tak masivnímu "on-the-fly" přepočítávání při každém požadavku na zobrazení doporučení. Zde si tento koncept jen znázorňujeme.
    res.render('admin', { 
        message: { type: 'success', text: 'Successfully recalculated TF-IDF metrics and Vector Norms! (Simulated)' }
    });
});

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
