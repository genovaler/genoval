var Migrations = artifacts.require("./Migrations.sol");
var Genvoal = artifacts.require("./GenovalToken.sol");

module.exports = function(deployer) {
  //deployer.deploy(Migrations);
  deployer.deploy(Genvoal);
};
