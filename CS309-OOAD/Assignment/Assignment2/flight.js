function onClickAddFlight() {
    let flightNo = document.querySelector('form input[name="flight-no"]').value;
    let origin = document.querySelector('form input[name="from"]').value;
    let destination = document.querySelector('form input[name="to"]').value;
    const year = document.getElementById("year").selectedIndex;
    const month = document.getElementById("month").selectedIndex;
    const day = document.getElementById("day").selectedIndex;
    if (validateInput(flightNo, origin, destination, year, month, day)) {
        addRow()
    }
}

function validateInput(flightNo, origin, destination, year, month, day) {
    let flightNoRegex = new RegExp(/^[A-Z0-9]{2}\d{3,4}$/);
    let airportCodeRegex = new RegExp(/^[A-Z]{3}$/);
    if (!flightNoRegex.test(flightNo)) {
        alert("Invalid Flight No.");
        return false;
    }
    if (!airportCodeRegex.test(origin)) {
        alert("Invalid origin airport code.");
        return false;
    }
    if (!airportCodeRegex.test(destination)) {
        alert("Invalid destination airport code.");
        return false;
    }

    if (year === -1 || month === -1 || day === -1) {
        alert("Invalid date airport code." + day);
        return false;
    }
    return true;
}

function initial() {
    setYear();
}

function setYear() {
    let year = document.getElementById("year");
    year.innerHTML = "";
    year.options.add(new Option("--", null));
    for (let i = 2000; i <= 2020; i++) {
        year.options.add(new Option(i, i));
    }
}

function setMonth() {
    let month = document.getElementById("month");
    month.innerHTML = "";
    month.options.add(new Option("--", null));
    for (let i = 1; i <= 12; i++) {
        month.options.add(new Option(i, i));
    }
}

function setDay() {
    let year = document.getElementById("year").value;
    let month = document.getElementById("month").value;
    let day = document.getElementById("day");
    let data = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    // clear the items
    day.innerHTML = "";
    // add new items
    day.options.add(new Option("--", null));
    for (let i = 1; i <= data[month - 1]; i++) {
        day.options.add(new Option(i, i));
    }
    if (((year % 4 === 0 && year % 100 !== 0) || year % 400 === 0) && month === 2) {
        day.options.add(new Option(29, 29));
    }
}

function addRow() {
    let bodyObj = document.getElementById("tbody");
    if (!bodyObj) {
        alert("Body of Table not Exist!");
        return;
    }
    let year = document.getElementById("year").value;
    let month = document.getElementById("month").value;
    let day = document.getElementById("day").value;
    let dhour = document.getElementById("dhour").value;
    let dminute = document.getElementById("dminute").value;
    let ahour = document.getElementById("ahour").value;
    let aminute = document.getElementById("aminute").value;
    let rowCount = bodyObj.rows.length;
    let cellCount = bodyObj.rows[0].cells.length;
    bodyObj.style.display = ""; // display the tbody
    let newRow = bodyObj.insertRow(rowCount++);
    newRow.insertCell(0).innerHTML = document.forms[0]["flight-no"].value;
    newRow.insertCell(1).innerHTML = document.forms[0]["airline-company"].value;
    newRow.insertCell(2).innerHTML = document.forms[0].from.value;
    newRow.insertCell(3).innerHTML = document.forms[0].to.value;
    newRow.insertCell(4).innerHTML = year + "/" + month + "/" + day;
    newRow.insertCell(5).innerHTML = dhour + ":" + dminute;
    newRow.insertCell(6).innerHTML = ahour + ":" + aminute;
    newRow.insertCell(7).innerHTML = bodyObj.rows[0].cells[cellCount - 1].innerHTML; // copy the "delete" button
    bodyObj.rows[0].style.display = "none"; // hide first row
}

function removeRow(inputobj) {
    if (!inputobj) return;
    let parentTD = inputobj.parentNode;
    let parentTR = parentTD.parentNode;
    let parentTBODY = parentTR.parentNode;
    parentTBODY.removeChild(parentTR);
}