const fs = require("fs")
var path = require("path")

var realCounter = 1
var fakeCounter = 1

fs.readdirSync("images/real").forEach(fileName => {
    fs.renameSync("images/real/" + fileName, "images/real/" + realCounter + path.extname(fileName));
    realCounter++
});

fs.readdirSync("images/fake").forEach(fileName => {
    fs.renameSync("images/fake/" + fileName, "images/fake/" + fakeCounter + path.extname(fileName));
    fakeCounter++
});