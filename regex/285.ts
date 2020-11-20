import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*not))((?=.*enjoy)|(?=.*experience))", "i"),
	caseID: 285
} as Regex;