import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*meet))((?=.*requirement)|(expectation)|(needs))", "i"),
	expressionDont: new RegExp("", "i"),
	caseID: 290,
	name: "This service does not guarantee that it or the products obtained through it meet the users' expectations or requirements"
} as Regex;