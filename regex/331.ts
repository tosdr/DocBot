import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^(((?=.*date\:)(?=.*effective))|((?=.*Last modified\:)|(?=.*Last modified on)|(?=.*Date of Last Revision)|(?=.*page was last edited)|(?=.*Last updated)|(?=.*Last update\:)|(?=.*Last Revised)|(?=.*Version Date)))", "i"),
	expressionDont: new RegExp("", "i"),
	caseID: 331,
	name: "There is a date of the last update of the terms"
} as Regex;