var models = {
  'The Scream': 'the_scream',
  'Mosaic': 'mosaic',
  'La Muse': 'la_muse',
  'Wave': 'wave',
  'Feathers': 'feathers',
  'Composition VII': 'composition_vii',
  'Cubist': 'cubist',
  'Udnie': 'udnie'
};

var selectedModel = models[$('#sel1 option:selected').text()];

$(document).ready(function(){

  selectedModel = models[$('#sel1 option:selected').text()];
  $('#sample').html('<img class="img-responsive" src=./images/'+selectedModel+'.jpg>');

  $('#upload-btn').on('click', function(){
    $('#upload').click();
  });

  $('#upload').on('change', function(e){
    var ext = $('#upload').prop("files")[0]['name'].split('.')[1];
    if (!ext){
      $('#exist').removeClass('hidden');
      $('#fileselected').html('');
    } else {
      $('#exist').addClass('hidden');
      $('#ext').addClass('hidden');
      $('#fileselected').html($('#upload').prop("files")[0]['name']);
    }
  });

  $('#sel1').on('change', function(){
    selectedModel = models[$('#sel1 option:selected').text()];
    $('#sample').html('<img class="img-responsive" src=./images/'+selectedModel+'.jpg>');
  });

  $('#upload-submit').on('click', function() {
    var fileElement = document.getElementById('upload');
    var file = fileElement.files[0];
    var extension = file.name.split('.')[1];
    
    var formData = new FormData();
    formData.append('file', file);
    formData.append('model', selectedModel);
    formData.append('extension', extension);

    var request = new XMLHttpRequest();
    request.open("POST", "/api/upload/");

    request.onreadystatechange = function() {
      if (request.readyState = XMLHttpRequest.DONE) {
        var response = JSON.parse(request.responseText);

        window.location.href = '/image?key=' + response.fileKey;
      }
    };

    request.send(formData);


  });

  /* $('#upload-submit').on('click', function(){
    var files = $('#upload').get(0).files;

    if (files.length > 0){
      var fd = new FormData();

      for (var i = 0; i < files.length; i++){
        fd.append('uploads', files[i], files[i]);
      }

      $.ajax({
        url:'/api/upload',
        type: 'POST',
        data: fd,
        processData: false,
        contentType: false,
        success: function(data){
          console.log('Upload successful!');
          $('#imageToggle').click();
          $('#convertedImage').html('<img>' + data + '</img>');
        },
        error: function(req, res, err){
          console.log(err);
        }
      });
    } else {
      console.log("No file!");
      $('form .hidden').removeClass('hidden');
    }

  }); */
});