import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*notice)|(?=.*comment)|(?=.*notify))((?=.*30 days)|(?=.*60 days))", "i"),
	caseID: 124,
	name: "When the service wants to make a material change to its terms, users are notified at least 30 days in advance"
} as Regex;