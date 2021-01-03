import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*portability)|(?=.*copy of your information)|(?=.*copy of your personal))", "i"),
	expressionDont: new RegExp("", "i"),
	caseID: 184,
	name: "This service provides a way for you to export your data"
} as Regex;