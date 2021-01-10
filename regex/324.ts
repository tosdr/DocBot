import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^^((((?=.*not)|(?=.*no))((?=.*multiple)|(?=.*more than one))(?=.*account))|(?=.*one-account-per-user)|((?=.*only)(?=.*one)(?=.*account)))", "i"),
	expressionDont: new RegExp("", "i"),
	caseID: 324,
	name: "Service does not allow alternative accounts"
} as Regex;