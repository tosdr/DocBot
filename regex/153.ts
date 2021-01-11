import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*substantial harm)|((?=.*fines)(?=.*spammers)))", "mi"),
	expressionDont: new RegExp("", "i"),
	caseID: 153,
	name: "This service fines spammers"
} as Regex;