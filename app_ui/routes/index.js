var express = require('express');
var router = express.Router();

/*access controllers*/
var ctrlMain = require ('../controllers/main');

router.get('/', ctrlMain.index);
router.get('/about', ctrlMain.about);
router.get('/examples', ctrlMain.examples);
router.get('/contact', ctrlMain.contact);
router.get('/image', ctrlMain.image);

module.exports = router;
