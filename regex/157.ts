import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*throttle)|(?=.*reduce))((?=.*speed)|(?=.*bandwidth))"),
	caseID: 157
} as Regex;