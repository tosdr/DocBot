import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*multiple))((?=.*not))((?=.*account))", "i"),
	caseID: 324
} as Regex;