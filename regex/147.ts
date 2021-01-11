import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*survive))((?=.*termination))", "mi"),
	expressionDont: new RegExp("", "i"),
	caseID: 147,
	name: "defend, indemnify, hold harmless; survives termination"
} as Regex;