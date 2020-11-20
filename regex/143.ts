import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*individual)|(?=.*personal))((?=.*non\-commercial)|(?=.*noncommercial))", "i"),
	caseID: 143
} as Regex;