function upload_image() {
    // show loading state when processing
    var loaders = document.getElementsByClassName("loader");
    for (var i = 0; i < loaders.length; i++) {
        loaders[i].style.display = "block";
    }
    var formGroups = document.getElementsByClassName("my-image-form");
    for (var j = 0; j < formGroups.length; j++) {
        formGroups[j].style.display = "none";
    }
}

function open_browser() {
    // open file browser
    $('#image_file').trigger('click');
}

function hide_know_more() {
    // when "No" is clicked on the result page
    var el = document.getElementById("know_more");
    if (el) {
        el.style.display = "none";
    }
}

$(document).ready(function () {
    $("input[type=file]").on('change', function () {
        // when a file is selected, clear the URL field and show the file name
        var urlInput = document.getElementById("image_url");
        if (urlInput) {
            urlInput.value = "";
        }
        var nameLabel = document.getElementById("imageName");
        if (nameLabel && this.files[0]) {
            nameLabel.style.display = "block";
            nameLabel.innerText = this.files[0].name;
        }
    });
});
