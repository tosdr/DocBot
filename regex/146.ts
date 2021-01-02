import { Regex } from '../models';

module.exports = {
    expression: new RegExp("^((?=.*indemnify))((?=.*defend))((?=.*harmless))", "i"),
	caseID: 146,
	name: "You agree to defend, indemnify, and hold the service harmless in case of a claim related to your use of the service"
} as Regex;