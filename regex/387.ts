import { Regex } from '../models';

module.exports = {
    expression: new RegExp("^((?=.*refer))((?=.*page)|(?=.*site))", "i"),
	expressionDont: new RegExp("", "i"),
	caseID: 387,
	name: "This service tracks which web page referred you to it"
} as Regex;