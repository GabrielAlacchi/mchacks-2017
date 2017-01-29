
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
      $('#original').removeClass('centered').attr('src', val.fileUrl);
    }
    if (val.completionUrl) {
      $('#filtered').removeClass('centered').attr('src', val.completionUrl);
      $('#download').removeClass('hidden').attr('href', val.completionUrl);
    }
    if (val.error){
      $('#filtered').attr('src', '');
      $('#imgerror').removeClass('hidden').html(val.error);
    }

  }); 
});
