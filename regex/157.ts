import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*throttle)|(?=.*reduce))((?=.*bandwidth)|(?=.*speed))", "i"),
	expressionDont: new RegExp("", "i"),
	caseID: 157,
	name: "This service throttles your use"
} as Regex;