import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*terminate)|(?=.*suspend)|(?=.*close))((?=.*account)|(?=.*membership)|(?=.*access))", "i"),
	caseID: 201,
	name: "The service can delete your account without prior notice and without a reason"
} as Regex;