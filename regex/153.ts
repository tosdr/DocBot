import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*substantial harm)|((?=.*fines)(?=.*spammers)))", "i"),
	expressionDont: new RegExp("", "i"),
	caseID: 153,
	name: "This service fines spammers"
} as Regex;