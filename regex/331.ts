import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*date\:)(?=.*effective))|((?=.*last modified)|(?=.*updated\:))|((?=.*update\:)|(?=.*last))|((?=.*Date of Last Revision)|(?=.*page was last edited)|(?=.*Last updated)|(?=.*Last Revised)(?=.*Version Date))", "i"),
	expressionDont: new RegExp("", "i"),
	caseID: 331,
	name: "There is a date of the last update of the terms"
} as Regex;