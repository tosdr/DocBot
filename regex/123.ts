import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*notice)|(?=.*comment)|(?=.*notify))((?=.*7 days)|(?=.*14 days))", "mi"),
	expressionDont: new RegExp("", "i"),
	caseID: 123,
	name: "When the service wants to change its terms, users are notified a week or more in advance"
} as Regex;