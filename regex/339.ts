import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*binding))((?=.*arbitration))", "i"),
	expressionDont: new RegExp("", "i"),
	caseID: 339,
	name: "This service forces users into binding arbitration in the case of disputes"
} as Regex;