import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*survive))((?=.*termination))", "i"),
	expressionDont: new RegExp("", "i"),
	caseID: 147,
	name: "defend, indemnify, hold harmless; survives termination"
} as Regex;