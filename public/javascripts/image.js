
function getParameterByName(name, url) {
  if (!url) {
    url = window.location.href;
  }
  name = name.replace(/[\[\]]/g, "\\$&");
  var regex = new RegExp("[?&]" + name + "(=([^&#]*)|&|#|$)"),
    results = regex.exec(url);
  if (!results) return null;
  if (!results[2]) return '';
  return decodeURIComponent(results[2].replace(/\+/g, " "));
}

$(function() {

  var db = firebase.database();
  var uploadKey = getParameterByName('key');

  db.ref('/uploads/' + uploadKey).on('value', function(snapshot) {

    var val = snapshot.val();
    if (val.fileUrl) {
      $('#original').attr('src', val.fileUrl);
    }
    if (val.completionUrl) {
      $('#filtered').attr('src', val.completionUrl);
    }

  });
});
