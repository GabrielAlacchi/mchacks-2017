var express = require('express');
var path = require('path');
var favicon = require('static-favicon');
var logger = require('morgan');
var cookieParser = require('cookie-parser');
var bodyParser = require('body-parser');
var firebase = require('firebase');

var routes = require('./app_ui/routes/index');
var routesApi = require('./app_api/routes');
var users = require('./app_ui/routes/users');

var app = express();

//firebase
var config = {
    apiKey: "AIzaSyDzEpeumcCznN6ICzjLSiRfMCUBfRqF_q0",
    authDomain: "mchacks-2017.firebaseapp.com",
    databaseURL: "https://mchacks-2017.firebaseio.com",
    storageBucket: "mchacks-2017.appspot.com",
    messagingSenderId: "152582265055"
};
firebase.initializeApp(config);

// view engine setup
app.set('views', path.join(__dirname, 'app_ui', 'views'));
app.set('view engine', 'jade');

app.use(favicon());
app.use(logger('dev'));
app.use(bodyParser.json());
app.use(bodyParser.urlencoded());
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));

app.use('/', routes);
app.use('/api', routesApi);
app.use('/users', users);

/// catch 404 and forwarding to error handler
app.use(function(req, res, next) {
    var err = new Error('Not Found');
    err.status = 404;
    next(err);
});

/// error handlers

// development error handler
// will print stacktrace
if (app.get('env') === 'development') {
    app.use(function(err, req, res, next) {
        res.status(err.status || 500);
        res.render('error', {
            message: err.message,
            error: err
        });
    });
}

// production error handler
// no stacktraces leaked to user
app.use(function(err, req, res, next) {
    res.status(err.status || 500);
    res.render('error', {
        message: err.message,
        error: {}
    });
});

module.exports = app;

